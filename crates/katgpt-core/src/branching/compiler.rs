//! `BudgetCompiler` — priority-cascade context compiler under a fixed byte
//! budget (Plan 329 T2.2).
//!
//! Assembles a [`CompiledContext`] from a heterogeneous set of retrieved
//! materials (scoped context, branch-local positive/negative memory,
//! cross-branch positive transfers, working memory, query). When the total
//! size exceeds `budget_bytes`, lower-priority materials are dropped first.
//!
//! # Priority cascade (RIZZ §"context compiler")
//!
//! Materials are admitted in this priority order (highest first):
//!
//! 1. **Scoped context** (`scope_ctx`) — task-family-level static context that
//!    scopes the entire generation. NEVER dropped before any other material.
//! 2. **Procedural rules** — branch-local IF-THEN heuristics with credit.
//! 3. **Episodic memory** — verifier-approved positive examples.
//! 4. **Cross-branch positive transfers** — helpful examples from sibling
//!    branches (RIZZ §"positive transfers").
//! 5. **Failure anti-patterns** — branch-local "do not do this" anchors.
//! 6. **Working memory** — recent scratch state.
//! 7. **Query** — the input prompt itself.
//!
//! Within each tier, items are admitted in caller-supplied order. When a tier
//! would overflow the remaining budget, items are dropped until the tier fits
//! (we never partially admit an item — atomic admission).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `CompiledContext` payload `E` | **Caller-defined** | Caller decides |
//! | `bytes_used`, `n_items` | **Raw** | YES (deterministic counters) |
//! | `budget_bytes` | **Raw** | YES (config) |
//! | Item priority tier | **Raw** | YES (deterministic enum) |
//!
//! The compiled context is consumed locally by the inference call that
//! triggered retrieval. It is NOT synced — it's a transient assembly.
//!
//! # Allocation discipline
//!
//! `compile()` writes into the pre-allocated `Vec`s inside [`CompiledContext`],
//! using `clear()` + reuse across calls. Pass the same `CompiledContext` to
//! successive `compile()` calls to amortize the allocation to zero after the
//! first call. The hot path is the admission loop, which is a single linear
//! scan with byte-accumulator + atomic-admission guard.

use crate::branching::types::{EpisodicEntry, FailureEntry, ProceduralRule};

/// Default byte budget for the compiled context. Generously sized for a typical
/// NPC cognition call (~4 KiB ≈ 1 page).
pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024;

/// Priority tier for a compiled item (higher = more important).
///
/// The cascade admits tiers top-down: `ScopeCtx` first, `Query` last. Lower
/// tiers are dropped first when the budget overflows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PriorityTier {
    /// The query / input prompt itself. Lowest priority — it's always
    /// re-derivable from the caller.
    Query = 0,
    /// Working memory (recent scratch state).
    WorkingMemory = 1,
    /// Failure anti-patterns (branch-local negative memory).
    Failures = 2,
    /// Cross-branch positive transfers (helpful examples from siblings).
    CrossBranchPositive = 3,
    /// Branch-local episodic memory (verifier-approved positive examples).
    Episodic = 4,
    /// Branch-local procedural rules (IF-THEN with credit).
    Procedural = 5,
    /// Scoped context (task-family-level static context). NEVER dropped first.
    ScopeCtx = 6,
}

impl PriorityTier {
    /// Human-readable name (for debugging / compiled-context dumps).
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::WorkingMemory => "working_memory",
            Self::Failures => "failures",
            Self::CrossBranchPositive => "cross_branch_positive",
            Self::Episodic => "episodic",
            Self::Procedural => "procedural",
            Self::ScopeCtx => "scope_ctx",
        }
    }
}

/// A single compiled item with its priority tier and byte cost.
///
/// `payload` is the caller-defined content (typed by `E`). `bytes` is the
/// caller-supplied serialized size of the payload — used by the budget gate.
/// The compiler never inspects payload contents; it only sums `bytes`.
#[derive(Clone, Debug)]
pub struct CompiledItem<E> {
    /// Priority tier (drives admission / drop order).
    pub tier: PriorityTier,
    /// Caller-supplied serialized size (bytes).
    pub bytes: usize,
    /// Caller-defined payload.
    pub payload: E,
}

impl<E> CompiledItem<E> {
    /// Construct a new compiled item.
    #[inline]
    #[must_use]
    pub fn new(tier: PriorityTier, bytes: usize, payload: E) -> Self {
        Self {
            tier,
            bytes,
            payload,
        }
    }
}

/// Heterogeneous retrieved materials ready for compilation.
///
/// The caller populates this from a retrieval pass (branch-local memory,
/// cross-branch positive transfers, working memory, query), then hands it to
/// [`BudgetCompiler::compile`]. All fields are `Vec`s so the caller can build
/// them incrementally; `compile()` consumes them by reference (no clone).
///
/// # Generic parameters
///
/// - `E` — the payload type for episodic entries and positive transfers.
/// - `F` — the payload type for failure entries (may differ from `E`).
/// - `W` — the payload type for working memory items.
/// - `Q` — the payload type for the query.
///
/// The scope context is a single optional `S` payload (not a `Vec`) because
/// it is task-family-level static context, not a list of retrieved items.
#[derive(Clone, Debug, Default)]
pub struct RetrievedMaterials<E, F = E, W = E, Q = E, S = E> {
    /// Scoped context (task-family-level static). Highest priority; admitted
    /// atomically as a single item.
    pub scope_ctx: Option<(usize, S)>,
    /// Branch-local procedural rules (with credit counters).
    pub procedural: Vec<ProceduralRule>,
    /// Branch-local episodic memory (verifier-approved positives).
    pub episodic: Vec<EpisodicEntry<E>>,
    /// Cross-branch positive transfers (helpful examples from siblings).
    pub cross_branch_positive: Vec<E>,
    /// Caller-supplied byte cost for each `cross_branch_positive` item
    /// (parallel array; same length as `cross_branch_positive`).
    pub cross_branch_bytes: Vec<usize>,
    /// Branch-local failure anti-patterns.
    pub failures: Vec<FailureEntry<F>>,
    /// Working memory items.
    pub working_memory: Vec<W>,
    /// Caller-supplied byte cost for each `working_memory` item (parallel
    /// array; same length as `working_memory`).
    pub working_memory_bytes: Vec<usize>,
    /// The query payload.
    pub query: Option<Q>,
    /// Caller-supplied byte cost for the query.
    pub query_bytes: usize,
}

/// Compiled context — the bounded output of [`BudgetCompiler::compile`].
///
/// Pre-allocated `Vec`s intended for `clear()` + reuse across successive
/// `compile()` calls. The `bytes_used` field is the authoritative total;
/// individual item bytes are not retained (only the payload and tier).
#[derive(Clone, Debug)]
pub struct CompiledContext<E> {
    /// Admitted items in admission order (highest tier first within each tier,
    /// then caller-supplied order within a tier).
    pub items: Vec<CompiledItem<E>>,
    /// Total bytes consumed by admitted items. Always ≤ `budget_bytes`.
    pub bytes_used: usize,
    /// The budget the compiler was configured with.
    pub budget_bytes: usize,
    /// Per-tier count of admitted items (indexed by `PriorityTier as usize`).
    pub tier_counts: [u32; 7],
    /// Per-tier count of dropped items (overflow).
    pub tier_dropped: [u32; 7],
}

impl<E> Default for CompiledContext<E> {
    #[inline]
    fn default() -> Self {
        Self {
            items: Vec::new(),
            bytes_used: 0,
            budget_bytes: DEFAULT_BUDGET_BYTES,
            tier_counts: [0u32; 7],
            tier_dropped: [0u32; 7],
        }
    }
}

impl<E> CompiledContext<E> {
    /// Construct with a pre-allocated items capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(item_capacity: usize, budget_bytes: usize) -> Self {
        Self {
            items: Vec::with_capacity(item_capacity),
            bytes_used: 0,
            budget_bytes,
            tier_counts: [0u32; 7],
            tier_dropped: [0u32; 7],
        }
    }

    /// Reset for reuse: clears items, zeroes counters, preserves capacity and
    /// budget. Call between successive `compile()` invocations.
    #[inline]
    pub fn reset(&mut self) {
        self.items.clear();
        self.bytes_used = 0;
        self.tier_counts = [0u32; 7];
        self.tier_dropped = [0u32; 7];
    }

    /// Total items admitted across all tiers.
    #[inline]
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.items.len()
    }

    /// True if no items were admitted (everything dropped or empty input).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Count admitted for a specific tier.
    #[inline]
    #[must_use]
    pub fn tier_count(&self, tier: PriorityTier) -> u32 {
        self.tier_counts[tier as usize]
    }

    /// Count dropped for a specific tier (overflow).
    #[inline]
    #[must_use]
    pub fn tier_dropped(&self, tier: PriorityTier) -> u32 {
        self.tier_dropped[tier as usize]
    }

    /// True if the budget was respected.
    #[inline]
    #[must_use]
    pub fn within_budget(&self) -> bool {
        self.bytes_used <= self.budget_bytes
    }
}

/// Priority-cascade context compiler under a fixed byte budget.
///
/// Construct with [`BudgetCompiler::new`] for a custom budget, or
/// [`BudgetCompiler::default`] for [`DEFAULT_BUDGET_BYTES`].
#[derive(Clone, Copy, Debug)]
pub struct BudgetCompiler {
    /// Maximum total bytes the compiled context may consume.
    pub budget_bytes: usize,
}

impl Default for BudgetCompiler {
    #[inline]
    fn default() -> Self {
        Self {
            budget_bytes: DEFAULT_BUDGET_BYTES,
        }
    }
}

impl BudgetCompiler {
    /// Construct with a custom byte budget.
    #[inline]
    #[must_use]
    pub const fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes }
    }

    /// Compile `materials` into `out`, applying the priority cascade.
    ///
    /// Resets `out` first (preserving its allocated capacity for reuse), then
    /// admits materials tier-by-tier (highest first). Within a tier, items
    /// are admitted in caller-supplied order until the tier is exhausted or
    /// the budget cannot fit the next item (atomic admission — never partial).
    ///
    /// After return:
    /// - `out.bytes_used <= self.budget_bytes` (always).
    /// - `out.tier_dropped[t]` counts items that didn't fit for tier `t`.
    /// - `out.items` is sorted by descending tier then caller-supplied order.
    ///
    /// # Allocation
    ///
    /// Allocates only when `out.items` needs to grow beyond its existing
    /// capacity. Steady-state (repeated `compile` into the same `out`) is
    /// zero-allocation.
    ///
    /// # Generic mapping
    ///
    /// The materials carry heterogeneous payload types (`E`, `F`, `W`, `Q`,
    /// `S`). The caller must provide mapping closures that convert each into
    /// the unified output payload type `O`, and report the per-item byte cost
    /// for the heterogeneous tiers where the compiler can't infer it.
    ///
    /// To keep the API tractable we use a single `byte_cost` closure applied
    /// uniformly; the caller encodes tier-specific cost logic inside it if
    /// needed. For the homogeneous tiers (`episodic`, `failures`,
    /// `cross_branch_positive`, `working_memory`) the byte cost comes from the
    /// parallel arrays or fixed estimates.
    #[allow(unused_assignments)] // `remaining` is decremented by `admit!` on the
                                 // final successful admission; the value is not
                                 // read afterward because the loop is over.
    pub fn compile<E, F, W, Q, S, O>(
        &self,
        materials: &RetrievedMaterials<E, F, W, Q, S>,
        out: &mut CompiledContext<O>,
        // Per-item byte cost for the scope_ctx.
        scope_bytes: impl Fn(&S) -> usize,
        // Per-item byte cost for a procedural rule.
        procedural_bytes: impl Fn(&ProceduralRule) -> usize,
        // Per-item byte cost for an episodic entry.
        episodic_bytes: impl Fn(&EpisodicEntry<E>) -> usize,
        // Per-item byte cost for a cross-branch positive item.
        cross_bytes: impl Fn(&E) -> usize,
        // Per-item byte cost for a failure entry.
        failure_bytes: impl Fn(&FailureEntry<F>) -> usize,
        // Per-item byte cost for a working memory item.
        working_bytes: impl Fn(&W) -> usize,
        // Per-item byte cost for the query.
        query_bytes: impl Fn(&Q) -> usize,
        // Conversions into the unified output payload O.
        scope_to_o: impl Fn(S) -> O,
        procedural_to_o: impl Fn(&ProceduralRule) -> O,
        episodic_to_o: impl Fn(&EpisodicEntry<E>) -> O,
        cross_to_o: impl Fn(&E) -> O,
        failure_to_o: impl Fn(&FailureEntry<F>) -> O,
        working_to_o: impl Fn(&W) -> O,
        query_to_o: impl Fn(Q) -> O,
    ) where
        E: Clone,
        F: Clone,
        W: Clone,
        Q: Clone,
        S: Clone,
        O: Clone,
    {
        out.reset();
        out.budget_bytes = self.budget_bytes;
        let budget = self.budget_bytes;
        let mut remaining = budget;

        // Helper macro: try to admit one item; on success decrement remaining
        // and push; on overflow increment tier_dropped.
        macro_rules! admit {
            ($tier:expr, $bytes:expr, $payload:expr) => {{
                let tier = $tier;
                let bytes = $bytes;
                if bytes <= remaining {
                    remaining -= bytes;
                    out.items.push(CompiledItem::new(tier, bytes, $payload));
                    out.bytes_used += bytes;
                    out.tier_counts[tier as usize] =
                        out.tier_counts[tier as usize].saturating_add(1);
                    true
                } else {
                    out.tier_dropped[tier as usize] =
                        out.tier_dropped[tier as usize].saturating_add(1);
                    false
                }
            }};
        }

        // Tier 1: ScopeCtx (highest priority, single optional item).
        if let Some((precomputed_bytes, scope)) = &materials.scope_ctx {
            // Prefer the caller's pre-supplied byte count if non-zero, else
            // recompute via the closure.
            let b = if *precomputed_bytes > 0 {
                *precomputed_bytes
            } else {
                scope_bytes(scope)
            };
            admit!(PriorityTier::ScopeCtx, b, scope_to_o(scope.clone()));
        }

        // Tier 2: Procedural rules.
        for rule in &materials.procedural {
            let b = procedural_bytes(rule);
            admit!(PriorityTier::Procedural, b, procedural_to_o(rule));
        }

        // Tier 3: Episodic memory.
        for entry in &materials.episodic {
            let b = episodic_bytes(entry);
            admit!(PriorityTier::Episodic, b, episodic_to_o(entry));
        }

        // Tier 4: Cross-branch positive transfers.
        // Use the parallel-bytes array if it matches length; else compute.
        let use_xb_parallel = materials.cross_branch_bytes.len() == materials.cross_branch_positive.len();
        for (i, item) in materials.cross_branch_positive.iter().enumerate() {
            let b = if use_xb_parallel {
                materials.cross_branch_bytes[i]
            } else {
                cross_bytes(item)
            };
            admit!(PriorityTier::CrossBranchPositive, b, cross_to_o(item));
        }

        // Tier 5: Failures.
        for failure in &materials.failures {
            let b = failure_bytes(failure);
            admit!(PriorityTier::Failures, b, failure_to_o(failure));
        }

        // Tier 6: Working memory.
        let use_wm_parallel = materials.working_memory_bytes.len() == materials.working_memory.len();
        for (i, item) in materials.working_memory.iter().enumerate() {
            let b = if use_wm_parallel {
                materials.working_memory_bytes[i]
            } else {
                working_bytes(item)
            };
            admit!(PriorityTier::WorkingMemory, b, working_to_o(item));
        }

        // Tier 7: Query (lowest priority).
        if let Some(query) = &materials.query {
            let b = if materials.query_bytes > 0 {
                materials.query_bytes
            } else {
                query_bytes(query)
            };
            admit!(PriorityTier::Query, b, query_to_o(query.clone()));
        }

        // Post-condition: budget respected.
        debug_assert!(
            out.bytes_used <= budget,
            "BudgetCompiler overflow: {} > {}",
            out.bytes_used,
            budget
        );
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branching::types::{BranchId, EpisodicEntry, FailureEntry, ProceduralRule};

    fn empty_materials() -> RetrievedMaterials<&'static str> {
        RetrievedMaterials::default()
    }

    fn fixed_cost<T>(_t: &T) -> usize {
        100
    }

    /// Identity for by-value closure positions (`scope_to_o: Fn(S)`,
    /// `query_to_o: Fn(Q)`). Works for any `Copy` payload (e.g. `&'static str`).
    fn id_val<T: Copy>(t: T) -> T {
        t
    }

    /// Identity for by-reference closure positions (`cross_to_o: Fn(&E)`,
    /// `working_to_o: Fn(&W)`). Clones out from behind the reference.
    fn id_ref<T: Clone>(t: &T) -> T {
        t.clone()
    }

    #[test]
    fn priority_tier_ordering() {
        // Higher enum value = higher priority.
        assert!(PriorityTier::ScopeCtx > PriorityTier::Procedural);
        assert!(PriorityTier::Procedural > PriorityTier::Episodic);
        assert!(PriorityTier::Episodic > PriorityTier::CrossBranchPositive);
        assert!(PriorityTier::CrossBranchPositive > PriorityTier::Failures);
        assert!(PriorityTier::Failures > PriorityTier::WorkingMemory);
        assert!(PriorityTier::WorkingMemory > PriorityTier::Query);
    }

    #[test]
    fn priority_tier_as_str() {
        assert_eq!(PriorityTier::ScopeCtx.as_str(), "scope_ctx");
        assert_eq!(PriorityTier::Query.as_str(), "query");
    }

    #[test]
    fn compile_empty_materials_yields_empty_context() {
        let compiler = BudgetCompiler::default();
        let mut out = CompiledContext::<&'static str>::default();
        let mats = empty_materials();
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            id_val,
            |_| "rule",
            |_| "episodic",
            id_ref,
            |_| "failure",
            id_ref,
            id_val,
        );
        assert!(out.is_empty());
        assert_eq!(out.bytes_used, 0);
        assert!(out.within_budget());
    }

    #[test]
    fn compile_admits_scope_ctx_first() {
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.scope_ctx = Some((50, "task_family_42"));
        mats.query = Some("hello");
        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |q| q.len(),
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );
        // Both scope_ctx (5 bytes) and query (5 bytes) fit in 1024.
        assert_eq!(out.n_items(), 2);
        assert!(out.within_budget());
        // ScopeCtx admitted (highest priority).
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.tier_dropped(PriorityTier::ScopeCtx), 0);
        // Query admitted.
        assert_eq!(out.tier_count(PriorityTier::Query), 1);
    }

    #[test]
    fn compile_drops_query_before_scope_ctx_on_overflow() {
        // Budget large enough for either but not both. ScopeCtx wins.
        let compiler = BudgetCompiler::new(10);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.scope_ctx = Some((8, "abcdefgh")); // 8 bytes
        mats.query = Some("xyz1234567"); // 10 bytes — won't fit alongside scope_ctx.
        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |q| q.len(),
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );
        // ScopeCtx admitted (8 bytes), query dropped (would exceed 10 - 8 = 2 left).
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.tier_dropped(PriorityTier::ScopeCtx), 0);
        assert_eq!(out.tier_count(PriorityTier::Query), 0);
        assert_eq!(out.tier_dropped(PriorityTier::Query), 1);
        assert!(out.within_budget());
        assert_eq!(out.bytes_used, 8);
    }

    #[test]
    fn compile_priority_cascade_drops_lowest_first() {
        // Set up materials in every tier. Tight budget forces cascade drops.
        let compiler = BudgetCompiler::new(350);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.scope_ctx = Some((50, "scope_payload"));
        // 3 procedural rules × 100 bytes = 300.
        for _ in 0..3 {
            mats.procedural.push(ProceduralRule {
                direction: vec![],
                antecedent: [0u8; 32],
                strategy: [0u8; 32],
                helpful: 0,
                harmful: 0,
            });
        }
        // 2 episodic entries × 100 bytes = 200.
        for _ in 0..2 {
            mats.episodic.push(EpisodicEntry {
                embedding: vec![],
                payload: "ep",
                reward: 0.5,
                scope: None,
                tick: 0,
            });
        }
        // Query (lowest priority, will be dropped).
        mats.query = Some("q");

        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |q| q.len(),
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );

        // Budget = 350. Costs: scope=50, procedural=300 (3×100), episodic=200 (2×100).
        // Admission order: scope (50, remaining 300), procedural (300, remaining 0),
        // episodic (200 won't fit → drop both), query (1 byte, but remaining 0 → drop).
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.tier_count(PriorityTier::Procedural), 3);
        assert_eq!(out.tier_dropped(PriorityTier::Episodic), 2);
        assert_eq!(out.tier_dropped(PriorityTier::Query), 1);
        assert_eq!(out.bytes_used, 50 + 300);
        assert!(out.within_budget());
    }

    #[test]
    fn compile_atomic_admission_never_partially_admits() {
        // An item larger than the entire budget is dropped, not truncated.
        let compiler = BudgetCompiler::new(10);
        let mut out = CompiledContext::<String>::default();
        let big: String = "x".repeat(100);
        let mut mats = RetrievedMaterials::<String, String, String, String, String>::default();
        // Pre-supply a byte count of 100 — larger than the 10-byte budget.
        mats.scope_ctx = Some((100, big));
        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |s| s.clone(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.clone(),
            |_| "failure".to_string(),
            |s| s.clone(),
            |s| s.clone(),
        );
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 0);
        assert_eq!(out.tier_dropped(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.bytes_used, 0);
    }

    #[test]
    fn compile_respects_budget_invariant() {
        // Stress: many items, small budget. Result must always satisfy
        // bytes_used ≤ budget.
        let compiler = BudgetCompiler::new(50);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        for _ in 0..100 {
            mats.episodic.push(EpisodicEntry {
                embedding: vec![],
                payload: "e",
                reward: 0.5,
                scope: None,
                tick: 0,
            });
        }
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            |_| 30, // 30 bytes per episodic entry
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );
        assert!(out.within_budget());
        // Budget 50, each item 30 → 1 admitted, 99 dropped.
        assert_eq!(out.tier_count(PriorityTier::Episodic), 1);
        assert_eq!(out.tier_dropped(PriorityTier::Episodic), 99);
        assert_eq!(out.bytes_used, 30);
    }

    #[test]
    fn compile_reuse_out_resets_state() {
        // Two successive compiles into the same `out` should not accumulate.
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<String>::default();
        let mut mats1 = RetrievedMaterials::<&'static str>::default();
        mats1.scope_ctx = Some((10, "first"));
        let mut mats2 = RetrievedMaterials::<&'static str>::default();
        mats2.query = Some("second");

        let build = |mats: &RetrievedMaterials<&'static str>, out: &mut CompiledContext<String>| {
            compiler.compile(
                mats,
                out,
                |s| s.len(),
                fixed_cost,
                fixed_cost,
                fixed_cost,
                fixed_cost,
                fixed_cost,
                |q| q.len(),
                |s| s.to_string(),
                |_| "rule".to_string(),
                |_| "episodic".to_string(),
                |s| s.to_string(),
                |_| "failure".to_string(),
                |s| s.to_string(),
                |q| q.to_string(),
            );
        };

        build(&mats1, &mut out);
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.tier_count(PriorityTier::Query), 0);
        assert_eq!(out.n_items(), 1);

        build(&mats2, &mut out);
        // Reset cleared the previous scope_ctx admission.
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 0);
        assert_eq!(out.tier_count(PriorityTier::Query), 1);
        assert_eq!(out.n_items(), 1);
    }

    #[test]
    fn compile_uses_parallel_byte_arrays_when_lengths_match() {
        // cross_branch_bytes parallel array overrides the closure.
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<&'static str>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.cross_branch_positive = vec!["a", "b", "c"];
        mats.cross_branch_bytes = vec![10, 20, 30]; // overrides closure
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |_| 999, // closure would give 999 — must NOT be used
            fixed_cost,
            fixed_cost,
            fixed_cost,
            id_val,
            |_| "rule",
            |_| "episodic",
            id_ref,
            |_| "failure",
            id_ref,
            id_val,
        );
        // Total = 10 + 20 + 30 = 60 bytes from parallel array.
        assert_eq!(out.bytes_used, 60);
        assert_eq!(out.tier_count(PriorityTier::CrossBranchPositive), 3);
    }

    #[test]
    fn compile_falls_back_to_closure_when_parallel_lengths_mismatch() {
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<&'static str>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.cross_branch_positive = vec!["a", "b", "c"];
        mats.cross_branch_bytes = vec![10, 20]; // mismatched length (2 vs 3)
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |_| 50, // closure fallback
            fixed_cost,
            fixed_cost,
            fixed_cost,
            id_val,
            |_| "rule",
            |_| "episodic",
            id_ref,
            |_| "failure",
            id_ref,
            id_val,
        );
        // Mismatched parallel → use closure: 3 × 50 = 150 bytes.
        assert_eq!(out.bytes_used, 150);
    }

    #[test]
    fn compiled_context_default_state() {
        let ctx = CompiledContext::<()>::default();
        assert_eq!(ctx.budget_bytes, DEFAULT_BUDGET_BYTES);
        assert_eq!(ctx.bytes_used, 0);
        assert!(ctx.is_empty());
        assert!(ctx.within_budget());
        assert_eq!(ctx.n_items(), 0);
        for tier in [
            PriorityTier::ScopeCtx,
            PriorityTier::Procedural,
            PriorityTier::Episodic,
            PriorityTier::CrossBranchPositive,
            PriorityTier::Failures,
            PriorityTier::WorkingMemory,
            PriorityTier::Query,
        ] {
            assert_eq!(ctx.tier_count(tier), 0);
            assert_eq!(ctx.tier_dropped(tier), 0);
        }
    }

    #[test]
    fn compiled_context_reset_preserves_capacity_and_budget() {
        let mut ctx = CompiledContext::<&'static str>::with_capacity(8, 256);
        ctx.items.push(CompiledItem::new(PriorityTier::Query, 10, "x"));
        ctx.bytes_used = 10;
        ctx.tier_counts[PriorityTier::Query as usize] = 1;
        ctx.reset();
        assert!(ctx.is_empty());
        assert_eq!(ctx.bytes_used, 0);
        assert_eq!(ctx.tier_counts, [0u32; 7]);
        // Capacity preserved.
        assert!(ctx.items.capacity() >= 8);
        // Budget is not reset by `reset()` (set by compiler on each compile).
        // Default value stays until compiler overwrites it.
        assert_eq!(ctx.budget_bytes, 256);
    }

    #[test]
    fn budget_compiler_default_and_new() {
        let d = BudgetCompiler::default();
        assert_eq!(d.budget_bytes, DEFAULT_BUDGET_BYTES);
        let c = BudgetCompiler::new(777);
        assert_eq!(c.budget_bytes, 777);
    }

    #[test]
    fn retrieved_materials_default_is_empty() {
        let m = RetrievedMaterials::<()>::default();
        assert!(m.scope_ctx.is_none());
        assert!(m.procedural.is_empty());
        assert!(m.episodic.is_empty());
        assert!(m.cross_branch_positive.is_empty());
        assert!(m.failures.is_empty());
        assert!(m.working_memory.is_empty());
        assert!(m.query.is_none());
    }

    #[test]
    fn compile_preserves_caller_order_within_tier() {
        // Within a tier, items are admitted in caller-supplied order.
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<&'static str>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.episodic.push(EpisodicEntry {
            embedding: vec![],
            payload: "first",
            reward: 0.5,
            scope: None,
            tick: 1,
        });
        mats.episodic.push(EpisodicEntry {
            embedding: vec![],
            payload: "second",
            reward: 0.5,
            scope: None,
            tick: 2,
        });
        mats.episodic.push(EpisodicEntry {
            embedding: vec![],
            payload: "third",
            reward: 0.5,
            scope: None,
            tick: 3,
        });
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            |_| 10,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            id_val,
            |_| "rule",
            |e| e.payload,
            id_ref,
            |_| "failure",
            id_ref,
            id_val,
        );
        // All three admitted, in order.
        assert_eq!(out.n_items(), 3);
        assert_eq!(out.items[0].payload, "first");
        assert_eq!(out.items[1].payload, "second");
        assert_eq!(out.items[2].payload, "third");
        assert_eq!(out.tier_count(PriorityTier::Episodic), 3);
    }

    #[test]
    fn compile_handles_failure_entries_with_distinct_payload_type() {
        // Failures use a distinct payload type F.
        let compiler = BudgetCompiler::new(1024);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str, u32>::default();
        mats.failures.push(FailureEntry {
            embedding: vec![],
            payload: 42u32,
            tick: 0,
        });
        compiler.compile(
            &mats,
            &mut out,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |_| 20,
            fixed_cost,
            fixed_cost,
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |f: &FailureEntry<u32>| format!("fail:{}", f.payload),
            |s| s.to_string(),
            |q| q.to_string(),
        );
        assert_eq!(out.tier_count(PriorityTier::Failures), 1);
        assert_eq!(out.items[0].payload, "fail:42");
        assert_eq!(out.bytes_used, 20);
    }

    #[test]
    fn scope_ctx_never_dropped_before_working_memory() {
        // Even with a tiny budget, scope_ctx wins over working memory because
        // it's admitted in an earlier tier.
        let compiler = BudgetCompiler::new(20);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.scope_ctx = Some((15, "scope")); // 15 bytes
        mats.working_memory.push("wm"); // 10 bytes via fixed_cost
        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );
        // scope_ctx (15) admitted; working memory (10) would overflow (15+10=25 > 20).
        assert_eq!(out.tier_count(PriorityTier::ScopeCtx), 1);
        assert_eq!(out.tier_dropped(PriorityTier::WorkingMemory), 1);
        assert_eq!(out.bytes_used, 15);
    }

    #[test]
    fn compile_admits_all_when_under_budget() {
        let compiler = BudgetCompiler::new(10000);
        let mut out = CompiledContext::<String>::default();
        let mut mats = RetrievedMaterials::<&'static str>::default();
        mats.scope_ctx = Some((10, "s"));
        mats.procedural.push(ProceduralRule {
            direction: vec![],
            antecedent: [0u8; 32],
            strategy: [0u8; 32],
            helpful: 0,
            harmful: 0,
        });
        mats.episodic.push(EpisodicEntry {
            embedding: vec![],
            payload: "e",
            reward: 0.5,
            scope: None,
            tick: 0,
        });
        mats.cross_branch_positive.push("x");
        mats.failures.push(FailureEntry {
            embedding: vec![],
            payload: "f",
            tick: 0,
        });
        mats.working_memory.push("w");
        mats.query = Some("q");

        compiler.compile(
            &mats,
            &mut out,
            |s| s.len(),
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            fixed_cost,
            |s| s.to_string(),
            |_| "rule".to_string(),
            |_| "episodic".to_string(),
            |s| s.to_string(),
            |_| "failure".to_string(),
            |s| s.to_string(),
            |q| q.to_string(),
        );

        // Every tier admits its 1 item.
        for tier in [
            PriorityTier::ScopeCtx,
            PriorityTier::Procedural,
            PriorityTier::Episodic,
            PriorityTier::CrossBranchPositive,
            PriorityTier::Failures,
            PriorityTier::WorkingMemory,
            PriorityTier::Query,
        ] {
            assert_eq!(out.tier_count(tier), 1, "tier {:?} should have 1 item", tier);
            assert_eq!(out.tier_dropped(tier), 0, "tier {:?} should have 0 dropped", tier);
        }
        assert_eq!(out.n_items(), 7);
        assert!(out.within_budget());
    }

    #[test]
    fn branch_id_unused_suppression() {
        // Sanity: ensure BranchId import compiles even if not used in asserts.
        let _ = BranchId::new(0);
    }
}
