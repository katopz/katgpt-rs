//! Shared traits for game AI and speculative decoding.
//!
//! Consolidated from katgpt-rs and riir-engine to eliminate duplication.
//! Both crates depend on `katgpt-core`, so moving traits here requires
//! zero new dependency edges.
//!
//! # Traits
//!
//! - [`ConstraintPruner`] — hard structural validity for DDTree branches
//! - [`ScreeningPruner`] — graded semantic relevance for speculative decoding
//! - [`GameState`] — forward model for what-if game simulation
//! - [`StateHeuristic`] — pluggable evaluation for non-terminal states
//! - [`RolloutPolicy`] — pluggable action selection for MCTS rollouts
//!
//! # Companion Structs
//!
//! - [`NoPruner`] — allows all tokens (baseline)
//! - [`BinaryScreeningPruner`] — adapter: ConstraintPruner → ScreeningPruner
//! - [`NoScreeningPruner`] — returns 1.0 for everything
//! - [`RandomRolloutPolicy`] — uniform random action selection
//! - [`ActionSpaceLog`] — per-tick branching factor metrics

use std::fmt;

use fastrand::Rng;

// ── ConstraintPruner ────────────────────────────────────────────

/// Trait for pruning drafted tokens against deterministic constraints.
///
/// The Deterministic Validator concept: before the target model verifies drafted
/// branches, a rules engine prunes invalid ones. This prevents the DDTree
/// from wasting budget on branches that can never be accepted.
///
/// Without pruner: DDTree explores ALL high-probability tokens.
/// With pruner:    DDTree explores only VALID high-probability tokens.
pub trait ConstraintPruner: Send + Sync {
    /// Check if `token_idx` at the given `depth` is valid, given the
    /// tokens placed at earlier depths in this path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    ///
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;

    /// Validate multiple token candidates at the same depth in a single call.
    ///
    /// Writes results into `results`: `results[i] = is_valid(depth, candidates[i], parent_tokens)`.
    /// Implementations can override this to amortize lock acquisition and setup costs
    /// across all candidates (e.g., single mutex lock + fuel reset for WASM).
    ///
    /// Default implementation calls `is_valid` per-item.
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = self.is_valid(depth, candidates[i], parent_tokens);
        }
    }

    /// Propagate semantic state with a new token. Returns true if all predicates hold.
    /// Default: no-op (delegates to is_valid for backward compat).
    fn propagate(&mut self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> bool {
        true // no-op by default
    }

    /// Soft validity score: how close is this token to the constraint boundary?
    /// Returns 1.0 for valid, 0.0 for invalid by default.
    /// Override for soft scoring (ManifoldE point-to-manifold, Plan 234).
    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.is_valid(depth, token_idx, parent_tokens) { 1.0 } else { 0.0 }
    }

    /// Returns the constraint as a half-space (normal vector, threshold) if available.
    /// Default: None (fall back to is_valid/manifold_score).
    fn constraint_vector(&self, _depth: usize, _parent_tokens: &[usize]) -> Option<(&[f32], f32)> {
        None
    }

    /// Sigmoid-graded reject confidence `P(reject) ∈ [0.0, 1.0]` (Plan 310 T1.2).
    ///
    /// `0.0` = definitely accept (valid token), `1.0` = definitely reject (invalid token),
    /// values in between = soft-reject candidate — caller may relax-and-retry instead of
    /// hard-failing. Distillation of HarnessBridge Table 7 (tolerant > strict rejection because
    /// false-reject cost > false-pass cost).
    ///
    /// **Default impl exactly reproduces today's binary behavior**: existing hard-reject maps
    /// to confidence `1.0`, accept maps to `0.0`. Every existing implementor is unchanged.
    ///
    /// **Sigmoid discipline (AGENTS.md):** new graded implementors MUST compute this as
    /// `sigmoid(β × evidence_strength)` — never softmax. Binary reject head → sigmoid.
    ///
    /// # Contract
    ///
    /// - Deterministic: same inputs → same output (no RNG).
    /// - Monotone in evidence strength for graded impls (no softmax crossover artifacts).
    /// - Range `[0.0, 1.0]` — callers clamp defensively but impls should not rely on it.
    #[inline]
    fn reject_confidence(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.is_valid(depth, token_idx, parent_tokens) { 0.0 } else { 1.0 }
    }

    /// Batch reject-confidence mirroring [`batch_is_valid`](Self::batch_is_valid)
    /// (Plan 310 T1.3).
    ///
    /// Writes results into `results`:
    /// `results[i] = reject_confidence(depth, candidates[i], parent_tokens)`.
    /// Implementations can override this to amortize lock acquisition and setup costs
    /// across all candidates (mirrors the [`batch_is_valid`](Self::batch_is_valid) pattern).
    ///
    /// Default implementation calls `reject_confidence` per-item.
    #[inline]
    fn batch_reject_confidence(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [f32],
    ) {
        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = self.reject_confidence(depth, candidates[i], parent_tokens);
        }
    }
}

/// No-op pruner: allows all tokens (original DDTree behavior).
pub struct NoPruner;

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        results[..len].fill(true);
    }

    fn propagate(&mut self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> bool {
        true
    }
}

impl DominoPruner for NoPruner {}

// ── DominoPruner (Plan 197, Research 177) ──────────────────────

/// Prefix-conditioned constraint correction for Domino causal correction.
///
/// Extends [`ConstraintPruner`] with a secondary correction pass that uses
/// the *specific prefix path* to refine validity decisions. This enables:
/// - **False positive elimination**: base says valid but prefix context says invalid
/// - **False negative recovery**: base says invalid but prefix context says valid (rare)
///
/// Default impl returns `base_valid` (no-op) — opt-in correction only.
///
/// # Implementors
///
/// - `SudokuPruner`: checks row/col/box constraints given the specific prefix path
/// - `SynPruner`: checks Rust syntax validity given preceding tokens
///   (e.g., `if` must be followed by `{` or condition)
pub trait DominoPruner: ConstraintPruner {
    /// Refine the base validity decision using prefix context.
    ///
    /// # Arguments
    /// * `depth` — current tree depth (position in the sequence)
    /// * `token` — candidate token index
    /// * `prefix` — tokens placed at earlier depths in this path
    /// * `base_valid` — result from the base [`ConstraintPruner::is_valid()`] check
    ///
    /// # Returns
    ///
    /// Corrected validity: `true` = keep, `false` = prune.
    #[inline]
    fn causal_correction(
        &self,
        _depth: usize,
        _token: usize,
        _prefix: &[usize],
        base_valid: bool,
    ) -> bool {
        base_valid
    }
}

// ── CompletionHorizon (Plan 207, Research 183) ──────────────────

/// Admissible "distance to a complete, valid output" for budget-aware pruning.
///
/// Extends [`ConstraintPruner`] with the **shortest-accepting-distance** `d(s)`:
/// a lower bound on how many *additional* tokens are needed, from the automaton
/// state reached after placing `token_idx` at `depth` (following `parent_tokens`),
/// to reach a complete & valid output. One precomputed integer per state powers
/// three things at once (Research 183):
///
/// - **(A) budget-aware masking** — prune any token whose successor cannot still
///   complete within the remaining token budget (TRUNCPROOF guarantee). This is
///   the property `build_dd_tree_lodestar` relies on.
/// - **(B) jump-ahead** — deterministic singular spans can be emitted in one step.
/// - **(C) A\* ordering + termination** — `d` is a monotone, admissible heuristic.
///
/// The novel equivalence (no single source states it): *min-completion-length ≡
/// remaining lattice height ≡ an admissible A\* heuristic.*
///
/// Default impls return `0` (no horizon info) so every existing
/// [`ConstraintPruner`] keeps working unchanged and pays **zero** overhead —
/// Lodestar is pure opt-in.
///
/// # Contract
///
/// `min_completion_distance` MUST be *admissible* (never overestimate the true
/// remaining length) for the budget guarantee to hold. Return [`u32::MAX`] to
/// signal "no valid completion reachable" — callers treat it as a hard prune.
pub trait CompletionHorizon: ConstraintPruner {
    /// Admissible lower bound on additional tokens needed to reach a complete,
    /// valid output *after* placing `token_idx` at `depth` following
    /// `parent_tokens` (the tokens at depths `0..depth`).
    ///
    /// Returns `0` by default (no horizon ⇒ budget masking is a no-op).
    #[inline]
    fn min_completion_distance(
        &self,
        _depth: usize,
        _token_idx: usize,
        _parent_tokens: &[usize],
    ) -> u32 {
        0
    }

    /// Length of the deterministic singular-path span starting at the state
    /// reached after `parent_tokens` — `0` means the next step is a real branch.
    /// Enables jump-ahead. Default `0`.
    #[inline]
    fn singular_span_len(&self, _depth: usize, _parent_tokens: &[usize]) -> u32 {
        0
    }
}

/// `NoPruner` has no horizon — budget masking is a no-op (returns 0).
impl CompletionHorizon for NoPruner {}

// ── FeatureClass (Plan 292 Phase 1, Research 267) ───────────────

/// Tags how a primitive reads model activations.
///
/// Distilled from Kortukov et al. 2026 (openreview 48NnVTsirb, Research 267 §1.1).
/// The empirical finding: features that *describe behavior already in the text*
/// (Detection) are a different linear subspace from features that *forecast the
/// probability of future behavior* (Prediction). Treating them as interchangeable
/// is the root cause of activation steering breaking outputs.
///
/// - **Detection** (`= 0`): reads features that *already* describe behavior in the
///   generated text (e.g. emotion vectors extracted from contrastive final-answer
///   pairs, CNA circuits). Safe for *monitoring* and for *intervention that
///   mutates behavior downstream of the read*. Risky to use as a *direct steering
///   target* — pushing activations along detection directions pushes the model
///   off-manifold.
///
/// - **Prediction** (`= 1`): reads features that *forecast* future behavior
///   probability from intermediate reasoning state (e.g. FPCG's future probe).
///   Safe as a *non-invasive steering target* via candidate selection (no
///   residual-stream modification).
///
/// All existing pruners inherit `Detection` by default (T1.2); the only
/// `Prediction`-side primitive currently in the tree is `FutureBehaviorProbe`
/// (Plan 292 Phase 2). The tag is a non-breaking addition — callers that don't
/// override the default are unchanged.
///
/// See `katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`
/// for the full empirical distinction and the FPCG algorithm that motivates it.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FeatureClass {
    /// Reads behavior already realized in the generated text (e.g. CAA / CNA
    /// / `EmotionDirections`). Default for all existing detection-side pruners.
    Detection = 0,
    /// Reads a linear forecast of future behavior probability from intermediate
    /// reasoning state (e.g. `FutureBehaviorProbe`, Plan 292 Phase 2).
    Prediction = 1,
}

// ── ScreeningPruner ─────────────────────────────────────────────

/// Graded relevance pruner replacing binary valid/invalid with continuous score.
///
/// Distilled from "Screening Is Enough" (arXiv:2604.01178).
/// Returns `R ∈ [0.0, 1.0]` which is blended into log-prob space:
/// - `1.0` = perfect match, no penalty (`ln(1.0) = 0.0`)
/// - `0.5` = mediocre match, soft penalty (`ln(0.5) ≈ -0.69`)
/// - `0.0` = hard rejection / trim (`ln(0.0) = -∞`)
///
/// This subsumes [`ConstraintPruner`] as the special case `R ∈ {0.0, 1.0}`.
/// Use [`BinaryScreeningPruner`] adapter to bridge between them.
///
/// # Ownership Boundary with ConstraintPruner (Plan 029, Task 7)
///
/// Single parser ownership: `ConstraintPruner` and `ScreeningPruner` make
/// **independent** decisions and must not compete for the same judgment:
///
/// - **`ConstraintPruner`** = hard structural validity (syntax, brackets, keywords).
///   Returns `bool`. Owns the decision: "is this token *syntactically* legal here?"
///
/// - **`ScreeningPruner`** = graded semantic relevance (domain fit, topic match).
///   Returns `f32` in `[0.0, 1.0]`. Owns the decision: "is this token *semantically*
///   relevant to the current domain?"
///
/// - **[`BinaryScreeningPruner`]** adapter = bridge only, zero additional logic.
///   Converts [`ConstraintPruner::is_valid()`] → `{0.0, 1.0}` relevance.
///
/// Both may prune the same token for different reasons — that's fine.
/// Both must NOT claim ownership of the same decision type — that's a bug.
pub trait ScreeningPruner: Send + Sync {
    /// Returns the absolute relevance of taking this token given the path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;

    /// How this primitive reads model activations (Plan 292 Phase 1, Research 267).
    ///
    /// Defaults to [`FeatureClass::Detection`] — every existing pruner in the
    /// tree inherits Detection, which matches the historical assumption that
    /// activation-reading primitives describe already-realized behavior.
    /// Primitives that instead forecast future behavior (e.g. `FutureBehaviorProbe`)
    /// override this to return [`FeatureClass::Prediction`]. The tag is purely
    /// advisory metadata for the screening-stack composer and `ReviewMetrics`
    /// telemetry — it does not change runtime semantics.
    ///
    /// Non-breaking: implementations that do not override are unchanged.
    #[inline]
    fn feature_class(&self) -> FeatureClass {
        FeatureClass::Detection
    }
}

/// Adapter: wraps any [`ConstraintPruner`] as a [`ScreeningPruner`] with binary relevance.
/// - `is_valid() == true` → relevance 1.0 (no penalty)
/// - `is_valid() == false` → relevance 0.0 (hard trim)
///
/// Use this to pass a [`ConstraintPruner`] where a [`ScreeningPruner`] is expected.
/// We use an explicit adapter instead of a blanket impl to avoid conflicts
/// with types that implement [`ConstraintPruner`] but need a custom [`ScreeningPruner`].
pub struct BinaryScreeningPruner<P>(pub P);

impl<P: ConstraintPruner + Send + Sync> ScreeningPruner for BinaryScreeningPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.0.is_valid(depth, token_idx, parent_tokens) { 1.0 } else { 0.0 }
    }
}

/// No-op screener: returns 1.0 for everything (no penalty, no trimming).
pub struct NoScreeningPruner;

impl ScreeningPruner for NoScreeningPruner {
    #[inline]
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

// ── Collapse-Aware Adaptive Thinking (Plan 212) ────────────────────

/// Monitors the token stream during reasoning and triggers early exit when
/// reasoning collapse is detected (hesitation patterns, repetitive tokens).
///
/// Modelless inference-time feature behind `collapse_aware_thinking` feature gate.
/// Integrates with `ThinkingController` to cut wasted budget on degenerate traces.
#[cfg(feature = "collapse_aware_thinking")]
pub trait CollapseDetector: Send + Sync {
    /// Returns `true` if the current trace exhibits collapse symptoms.
    fn check_collapse(&mut self, token_id: u32, position: usize) -> bool;

    /// Reset internal state between traces. May update self-tuning parameters.
    fn reset(&mut self);

    /// Number of hesitation tokens observed in the current trace.
    fn hesitation_count(&self) -> u32;

    /// Current collapse threshold τ.
    fn threshold(&self) -> u32;
}

// ── GameState ───────────────────────────────────────────────────

/// Forward model trait — any game state that supports what-if simulation.
///
/// Implementors must be cheaply cloneable snapshots (~KB, not MB).
/// The arena converts its internal state → snapshot once per tick,
/// then search algorithms work entirely on snapshots.
///
/// # Type Parameters
/// - `Action`: the move type for this game domain
///
/// # Required Methods
/// - `available_actions`: legal moves for a player
/// - `advance`: pure successor state (no mutation)
/// - `is_terminal`: game-over check
/// - `reward`: terminal value for a player
/// - `tick`: current turn number
pub trait GameState: Clone {
    /// Move type for this game domain (e.g., `BomberAction`, `fft::Action`).
    type Action: Clone;

    /// Legal actions for `player_id` in current state.
    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;

    /// Fill `buf` with legal actions for `player_id`, clearing it first.
    ///
    /// Default implementation calls [`available_actions()`](Self::available_actions)
    /// and moves items into `buf`. Override to avoid intermediate allocation.
    fn available_actions_into(&self, player_id: u8, buf: &mut Vec<Self::Action>) {
        buf.clear();
        buf.extend(self.available_actions(player_id));
    }

    /// Apply action, return successor state. Does NOT mutate `self`.
    ///
    /// Default implementation clones `self` then calls
    /// [`advance_inplace()`](Self::advance_inplace). Override `advance_inplace`
    /// (NOT this method) to make the clone-free path available to callers that
    /// can mutate in place (MCTS rollouts, tree descent).
    ///
    /// Callers that need a fresh owned successor (tree-node expansion, test
    /// assertions) use this. Callers that discard the parent state should call
    /// [`advance_inplace()`](Self::advance_inplace) instead.
    ///
    /// **Implementor contract (recursion trap):** every concrete impl MUST
    /// override at least one of `advance` / `advance_inplace`. The defaults
    /// delegate to each other; an impl that overrides NEITHER infinite-recurses
    /// (`advance` → `advance_inplace` → `advance` → ...). All existing impls
    /// define `advance`, so they are safe. Mirrors the
    /// `available_actions` / `available_actions_into` mutual-default pair.
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self {
        let mut next = self.clone();
        next.advance_inplace(action, player_id);
        next
    }

    /// Apply action, mutating `self` in place. The clone-free path.
    ///
    /// Default implementation delegates to [`advance()`](Self::advance) (which
    /// clones). Override this to avoid the allocation when `self` owns heap
    /// data (e.g. `Vec` fields like `FrameSnapshot::nearby_entities` /
    /// `threats`). Callers that discard the parent state (MCTS rollouts, path
    /// descent) should call this instead of `advance`.
    ///
    /// **Implementor contract (recursion trap):** see `advance` — every impl
    /// must override at least one of the pair.
    fn advance_inplace(&mut self, action: &Self::Action, player_id: u8) {
        *self = self.advance(action, player_id);
    }

    /// Is the game over?
    fn is_terminal(&self) -> bool;

    /// Terminal reward for `player_id` (higher = better, typically 0..1).
    fn reward(&self, player_id: u8) -> f32;

    /// Current tick/turn number.
    fn tick(&self) -> u32;

    /// Number of legal actions for `player_id`.
    ///
    /// **Override this** if you can compute the count without building the full action Vec.
    /// The default implementation calls [`available_actions()`](Self::available_actions)
    /// which allocates — expensive in tight MCTS loops.
    fn action_space_size(&self, player_id: u8) -> usize {
        self.available_actions(player_id).len()
    }
}

// ── StateHeuristic ──────────────────────────────────────────────

/// Pluggable heuristic for evaluating non-terminal states.
///
/// Used by search algorithms (MCTS rollouts, RHEA fitness) when
/// [`GameState::is_terminal()`] is false but we need a numeric evaluation.
///
/// Domain-specific heuristics beat generic search (STRATEGA finding),
/// so each game provides its own implementation.
pub trait StateHeuristic<S: GameState> {
    /// Evaluate state for `player_id`. Higher = better.
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}

// ── RolloutPolicy ───────────────────────────────────────────────

/// Pluggable rollout policy for MCTS.
///
/// Replaces hardcoded random selection with informed action choice.
/// The default [`RandomRolloutPolicy`] preserves existing behavior.
///
/// # Implementors
/// - [`RandomRolloutPolicy`]: uniform random (baseline)
/// - `BanditRolloutPolicy<S>`: ε-greedy guided by bandit Q-values (riir-engine)
pub trait RolloutPolicy<S: GameState> {
    /// Select an action index from `actions` during MCTS rollout.
    ///
    /// # Arguments
    /// * `state` — current rollout state
    /// * `actions` — available actions for `player_id`
    /// * `player_id` — which player is acting
    /// * `rng` — RNG for stochastic policies
    ///
    /// # Returns
    /// Index into `actions` (0..actions.len()).
    fn select(&mut self, state: &S, actions: &[S::Action], player_id: u8, rng: &mut Rng) -> usize;
}

/// Uniform random rollout policy — baseline, identical to original MCTS behavior.
///
/// Every action has equal probability. Use this as a control group when
/// comparing against informed rollout policies.
pub struct RandomRolloutPolicy;

impl<S: GameState> RolloutPolicy<S> for RandomRolloutPolicy {
    #[inline]
    fn select(
        &mut self,
        _state: &S,
        actions: &[S::Action],
        _player_id: u8,
        rng: &mut Rng,
    ) -> usize {
        rng.usize(0..actions.len())
    }
}

// ── ActionSpaceLog ──────────────────────────────────────────────

/// Per-tick action space metrics for branching factor analysis.
///
/// Tracks how the action space evolves across ticks — useful for:
/// - Validating search budget vs branching factor
/// - Detecting game phases (opening → midgame → endgame)
/// - Comparing action space across game domains
///
/// Per-player aggregate tracked during insert for O(1) reads.
#[derive(Clone, Copy, Debug, Default)]
struct PlayerAgg {
    // OPT: usize first avoids 4 bytes of padding between f64 and usize
    count: usize,
    // f64 (Issue 690): f32 running sums drift low once the total passes ~2^24
    // (one ULP exceeds 1) — measured −2.81% on a 2.26M-record go-arena arm.
    // f64 holds exact integer sums to 2^53, unreachable at any arena scale.
    sum: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ActionSpaceLog {
    /// Running peak across all entries, tracked during record() for O(1) peak_action_space().
    peak: usize,
    /// Running total sum of action counts for O(1) avg_action_space().
    /// f64 for exact integer accumulation past 2^24 (Issue 690: the f32 form
    /// under-reported the mean by −2.81% at 2.26M records × 50 actions).
    total_sum: f64,
    /// (tick, player_id, action_count) entries.
    /// Field order: usize (8B) → u32 (4B) → u8 (1B) = 16B vs 24B with (u32, u8, usize).
    entries: Vec<(usize, u32, u8)>,
    /// Per-player running aggregates for O(1) avg_action_space_for().
    /// Indexed by player_id (u8, so max 256 entries). Lazy-initialized on first record.
    player_aggs: Vec<PlayerAgg>,
}

impl ActionSpaceLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty log with pre-allocated capacity.
    /// Use this in hot paths (e.g., game rollouts) where the approximate
    /// number of entries is known upfront.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            player_aggs: Vec::new(),
            peak: 0,
            total_sum: 0.0,
        }
    }

    /// Record action space size for a player at the current tick.
    /// Maintains per-player aggregates for O(1) avg_action_space_for().
    pub fn record<S: GameState>(&mut self, state: &S, player_id: u8) {
        let n = state.action_space_size(player_id);
        let pid = player_id as usize;
        // Extend player_aggs if needed (at most 256 entries)
        if self.player_aggs.len() <= pid {
            self.player_aggs
                .resize(pid + 1, PlayerAgg { sum: 0.0, count: 0 });
        }
        self.player_aggs[pid].sum += n as f64;
        self.player_aggs[pid].count += 1;
        self.total_sum += n as f64;
        if n > self.peak {
            self.peak = n;
        }
        self.entries.push((n, state.tick(), player_id));
    }

    /// Total number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the log empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Average action space size across all entries.
    /// O(1) via running total_sum tracked during record(). Exact: the f64
    /// accumulator holds integer sums exactly to 2^53 (Issue 690).
    pub fn avg_action_space(&self) -> f32 {
        if self.entries.is_empty() {
            0.0
        } else {
            (self.total_sum / self.entries.len() as f64) as f32
        }
    }

    /// Average action space size for a specific player.
    /// O(1) via pre-tracked per-player aggregates — no linear scan.
    pub fn avg_action_space_for(&self, player_id: u8) -> f32 {
        let pid = player_id as usize;
        let agg = match self.player_aggs.get(pid) {
            Some(a) if a.count > 0 => a,
            _ => return 0.0,
        };
        (agg.sum / agg.count as f64) as f32
    }

    /// Peak (maximum) action space size recorded.
    /// O(1) via running peak tracked during record().
    #[inline]
    pub fn peak_action_space(&self) -> usize {
        self.peak
    }

    /// Clear all entries and reset per-player aggregates.
    pub fn clear(&mut self) {
        self.entries.clear();
        for agg in &mut self.player_aggs {
            agg.sum = 0.0;
            agg.count = 0;
        }
        self.peak = 0;
        self.total_sum = 0.0;
    }
}

impl fmt::Display for ActionSpaceLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.entries.is_empty() { write!(f, "ActionSpaceLog(empty)") } else { write!(
                f,
                "ActionSpaceLog(entries={}, avg={:.1}, peak={})",
                self.entries.len(),
                self.avg_action_space(),
                self.peak_action_space()
            ) }
    }
}

// ── LEO All-Goals Traits (Plan 155) ──────────────────────────────
//
// LEO (Learn Everything All at Once) outputs Q-values for ALL goals
// simultaneously instead of conditioning on a single goal (UVFA-style).
// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
//   All at Once", ICML 2026. arXiv:2605.23551
//   https://github.com/MichaelTMatthews/purejaxgcrl
//
// Feature gates:
//   leo_all_goals — LeoHead + AllGoalsUpdate + sigmoid_bounded_q
//   dual_leo      — + DualLeoMixer + AutocurriculumSampler

// ── LEO All-Goals Trait Framework (Plan 155) ────────────────────
//
// Architecture note — Batch Renormalization (BatchRenorm):
//
// The reference JAX implementation uses BatchRenorm (not standard BatchNorm
// or LayerNorm) for network stability with highly off-policy data. Key params:
//   - r_max = 3 (maximum correction ratio)
//   - d_max = 5 (maximum correction difference)
//   - warmup = 1000 steps (gradual correction ramp-up)
//
// BatchRenorm constrains the running-statistics correction to prevent
// divergence when training on highly off-policy replay data — critical
// for LEO's all-goals Q-learning where the same network processes
// experiences from many different goal-conditioned policies.
//
// Implementors SHOULD use BatchRenorm (or an equivalent constrained
// normalization) in their LeoHead network architectures. Standard
// BatchNorm is insufficient; LayerNorm is acceptable but may underperform.
//
// Ref: Ioffe (2017) "Batch Renormalization" — arXiv:1702.03275
// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
//   All at Once", ICML 2026. arXiv:2605.23551 — Section 5.1

/// Bound Q-value estimates with sigmoid to prevent divergence.
///
/// CRITICAL: Without this, LEO's Q-values frequently diverge due to
/// highly off-policy updates (paper Section 5.1).
///
/// Maps raw Q ∈ (-∞, +∞) → bounded Q ∈ (0, 1).
///
/// Delegates to shared crate::simd::fast_sigmoid (Cephes-exp accuracy, ~1 ULP).
#[cfg(feature = "leo_all_goals")]
#[inline]
pub fn sigmoid_bounded_q(raw_q: f32) -> f32 {
    crate::simd::fast_sigmoid(raw_q)
}

/// All-goals Q-value output head (LEO architecture).
///
/// Instead of conditioning on a goal (UVFA-style), this outputs Q-values
/// for ALL goals simultaneously: Q(s) → R^{G×A}.
///
/// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
///   All at Once", ICML 2026. arXiv:2605.23551
#[cfg(feature = "leo_all_goals")]
pub trait LeoHead {
    /// Compute Q-values for all goals × all actions from state.
    /// Returns `[goals * actions]` flattened (row-major: goal-major).
    fn all_goals_q(&self, state: &[f32]) -> Vec<f32>;

    /// Number of goals in the output head.
    fn goal_count(&self) -> usize;

    /// Number of discrete actions per goal.
    fn action_count(&self) -> usize;

    /// Extract Q-values for a specific goal by indexing into the flat output.
    fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
        let start = goal * self.action_count();
        &all_q[start..start + self.action_count()]
    }
}

/// Vectorized all-goals Bellman update.
///
/// L = (R(s') + γ · max_a' Q(a'|s') - Q(a|s))²
///
/// Where R(s') ∈ R^G is the reward vector across ALL goals.
/// Single forward pass updates all |G| Q-value heads simultaneously.
#[cfg(feature = "leo_all_goals")]
pub trait AllGoalsUpdate {
    /// Compute all-goals TD target.
    ///
    /// - `rewards`: `[goals]` — R(s', g) for all g
    /// - `next_q`: `[goals][actions]` — Q(s', a', g) for all g, a
    /// - Returns: `[goals]` — TD target per goal
    fn td_target(&self, rewards: &[f32], next_q: &[Vec<f32>], gamma: f32) -> Vec<f32> {
        rewards
            .iter()
            .zip(next_q.iter())
            .map(|(&r, q_next)| {
                // SIMD max reduction replaces scalar `fold(f32::NEG_INFINITY, f32::max)`.
                // Same semantics for non-NaN inputs (including empty -> -inf); vectorizes on NEON/AVX2.
                let max_q = crate::simd::simd_max_f32(q_next);
                r + gamma * max_q
            })
            .collect()
    }

    /// Compute all-goals TD loss (MSE) averaged across goals.
    ///
    /// - `predicted`: `[goals]` each containing chosen-action Q-values
    /// - `target`: `[goals]` TD targets
    fn loss(predicted: &[Vec<f32>], target: &[f32]) -> f32 {
        predicted
            .iter()
            .zip(target.iter())
            .map(|(q_pred, &q_tgt)| {
                let chosen = q_pred[0]; // first action as chosen (caller should index correctly)
                0.5 * (chosen - q_tgt).powi(2)
            })
            .sum::<f32>()
            / predicted.len().max(1) as f32
    }

    /// Compute all-goals Q(λ) TD target with eligibility traces.
    ///
    /// Multi-step TD with trace decay parameter λ:
    ///   G(g) = R(g) + γ · [λ · G_next(g) + (1-λ) · max_a' Q(s',a',g)] · (1-done(g))
    ///
    /// - `rewards`: `[goals]` — R(s', g) for all g
    /// - `next_q_max`: `[goals]` — max_a' Q(s', a', g) for all g (pre-computed)
    /// - `next_lambda_return`: `[goals]` — G_next(g) λ-return from next timestep (0 for last step)
    /// - `done`: `[goals]` — whether goal g is terminal
    /// - `gamma`: discount factor
    /// - `lambda`: trace decay (0 = one-step TD, 1 = Monte Carlo)
    /// - Returns: `[goals]` — Q(λ) target per goal
    fn td_target_lambda(
        &self,
        rewards: &[f32],
        next_q_max: &[f32],
        next_lambda_return: &[f32],
        done: &[bool],
        gamma: f32,
        lambda: f32,
    ) -> Vec<f32> {
        rewards
            .iter()
            .zip(next_q_max.iter())
            .zip(next_lambda_return.iter())
            .zip(done.iter())
            .map(|(((&r, &q_max), &g_next), &d)| if d { r } else { r + gamma * (lambda * g_next + (1.0 - lambda) * q_max) })
            .collect()
    }
}

/// Acting mode for dual LEO mixing.
///
/// Controls how LEO (teacher) and UVFA (student) Q-values are combined
/// for action selection. From JAX `DUAL_LEO_ACTING_MODE` config.
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActingMode {
    /// Linear combination: Q = (1-α)·Q_UVFA + α·Q_LEO.
    /// Default mode, sweep winner on Craftax.
    #[default]
    Lc,
    /// LEO-only ablation: Q = `Q_LEO[:,:,g]`.
    LeoOnly,
    /// UVFA-only ablation: Q = `Q_UVFA[:,g]`.
    UvfaOnly,
    /// Optimistic combining: Q = max(Q_LEO, Q_UVFA).
    Max,
    /// Pessimistic combining: Q = min(Q_LEO, Q_UVFA).
    Min,
}

/// Schedule for the mixing coefficient α over training.
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum AlphaSchedule {
    /// Constant α throughout training (default).
    /// Sweep uses 0.3 with `anneal_lc_leo=false`.
    Fixed(f32),
    /// Linearly anneal α from `start` to `end` over training.
    /// From JAX: coef = p * end + (1-p) * start, where p = step/total_steps.
    LinearAnneal { start: f32, end: f32 },
}

#[cfg(feature = "dual_leo")]
impl Default for AlphaSchedule {
    fn default() -> Self {
        AlphaSchedule::Fixed(0.3)
    }
}

/// Behavioral cloning target source for BC regularization (PPO variant).
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BcTarget {
    /// Follow LEO's greedy (argmax) action.
    #[default]
    Argmax,
}

/// Behavioral cloning regularization config (Dual LEO PPO variant).
///
/// The UVFA student's policy is regularized toward LEO's argmax action
/// early in training, then the BC coefficient decays to 0.
///
/// From `dual_leo_ppo.py`: bc_coef_policy=0.1, bc_coef_value=0.0,
/// bc_policy_target="argmax", anneal_bc=true.
#[cfg(feature = "dual_leo")]
#[derive(Clone, Copy, Debug)]
pub struct BcConfig {
    /// PPO policy regularization coefficient toward LEO's action.
    /// Default: 0.1
    pub policy_coef: f32,
    /// Value function BC coefficient. Default: 0.0 (disabled in sweep).
    pub value_coef: f32,
    /// Which action to use as BC target.
    pub target: BcTarget,
    /// Whether to anneal BC coefficient to 0 over training.
    pub anneal: bool,
}

#[cfg(feature = "dual_leo")]
impl Default for BcConfig {
    fn default() -> Self {
        Self {
            policy_coef: 0.1,
            value_coef: 0.0,
            target: BcTarget::Argmax,
            anneal: true,
        }
    }
}

/// Dual LEO mixing between teacher (LEO) and student (UVFA).
///
/// Q_combined(g) = α·Q_LEO(s,a,g) + (1-α)·Q_UVFA(s,a,g)
///
/// α controls modelless→model trust transfer:
/// - High α: trust LEO teacher (modelless, broad)
/// - Low α: trust UVFA student (model-based, precise)
#[cfg(feature = "dual_leo")]
pub trait DualLeoMixer {
    /// Mix LEO and UVFA Q-values for acting on a specific goal.
    fn mix(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32> {
        q_leo
            .iter()
            .zip(q_uvfa.iter())
            .map(|(&ql, &qu)| alpha * ql + (1.0 - alpha) * qu)
            .collect()
    }

    /// Mix LEO and UVFA Q-values into a pre-allocated buffer, avoiding allocation.
    fn mix_into(&self, out: &mut [f32], q_leo: &[f32], q_uvfa: &[f32], alpha: f32) {
        for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
            *o = alpha * ql + (1.0 - alpha) * qu;
        }
    }

    /// Default α = 0.3 (from paper sweep on Craftax).
    fn default_alpha(&self) -> f32 {
        0.3
    }

    /// Which acting mode to use. Default: Lc (sweep winner).
    fn acting_mode(&self) -> ActingMode {
        ActingMode::Lc
    }

    /// Alpha schedule over training. Default: Fixed(0.3).
    fn alpha_schedule(&self) -> AlphaSchedule {
        AlphaSchedule::Fixed(self.default_alpha())
    }

    /// Resolve alpha for the current training progress (0.0..=1.0).
    fn alpha_at_progress(&self, progress: f32) -> f32 {
        match self.alpha_schedule() {
            AlphaSchedule::Fixed(a) => a,
            AlphaSchedule::LinearAnneal { start, end } => {
                progress.clamp(0.0, 1.0) * end + (1.0 - progress.clamp(0.0, 1.0)) * start
            }
        }
    }

    /// Combine Q-values using the configured acting mode.
    fn combine(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; q_leo.len()];
        self.combine_into(&mut out, q_leo, q_uvfa, alpha);
        out
    }

    /// Zero-alloc variant of [`combine`](Self::combine): writes into a pre-allocated buffer.
    fn combine_into(&self, out: &mut [f32], q_leo: &[f32], q_uvfa: &[f32], alpha: f32) {
        match self.acting_mode() {
            ActingMode::Lc => self.mix_into(out, q_leo, q_uvfa, alpha),
            ActingMode::LeoOnly => out.copy_from_slice(q_leo),
            ActingMode::UvfaOnly => out.copy_from_slice(q_uvfa),
            ActingMode::Max => {
                for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
                    *o = ql.max(qu);
                }
            }
            ActingMode::Min => {
                for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
                    *o = ql.min(qu);
                }
            }
        }
    }

    /// BC regularization config (Dual LEO PPO variant). Default: no BC.
    fn bc_config(&self) -> Option<BcConfig> {
        None
    }
}

/// Goal sampling from previously observed goals only.
///
/// "We sample goals only from goals observed at least once in the past,
/// to prevent completely out-of-reach goals being sampled."
/// — Matthews et al. (2026), ICML 2026, arXiv:2605.23551
#[cfg(feature = "dual_leo")]
pub trait AutocurriculumSampler {
    /// Sample a goal uniformly from previously observed goals.
    fn sample_goal(&self, rng: &mut Rng) -> usize;

    /// Mark a goal as observed (first time seen in any trajectory).
    fn observe_goal(&mut self, goal: usize);

    /// Number of unique goals observed so far.
    fn observed_count(&self) -> usize;

    /// Total goals in the goal set.
    fn total_goal_count(&self) -> usize;

    /// Update observed goals from a batch of observations.
    ///
    /// Checks which goals match any observation in the batch (union matching).
    /// Returns the updated boolean mask over all goals.
    ///
    /// From JAX `get_goals_seen()`: a goal is "seen" if any obs in the batch
    /// matches the goal's observation pattern (match_sum > 0).
    ///
    /// - `obs_batch`: batch of observation vectors
    /// - `all_goals`: all goal observation patterns `[goals][features]`
    /// - `current_mask`: current `[goals]` boolean mask (true = seen)
    /// - Returns: updated `[goals]` boolean mask
    fn update_goals_seen(
        &self,
        obs_batch: &[Vec<f32>],
        all_goals: &[Vec<f32>],
        current_mask: &[bool],
    ) -> Vec<bool> {
        let mut mask = current_mask.to_vec();
        // Pre-compute goal norms once (avoid redundant recomputation per obs).
        // SIMD: sum-of-squares via simd_sum_sq (NEON/AVX2 FMA) instead of scalar map-sum.
        let norm_goals: Vec<f32> = all_goals
            .iter()
            .map(|goal_obs| crate::simd::simd_sum_sq(goal_obs, goal_obs.len()).sqrt())
            .collect();
        for obs in obs_batch {
            let norm_obs: f32 = crate::simd::simd_sum_sq(obs, obs.len()).sqrt();
            for (g, goal_obs) in all_goals.iter().enumerate() {
                if g < mask.len() && !mask[g] {
                    // Union matching: normalized cosine-like similarity.
                    // Threshold > 0.9 ensures only near-exact matches count.
                    // JAX uses binary match_sum > 0 on discretized observations.
                    let dot: f32 =
                        crate::simd::simd_dot_f32(obs, goal_obs, obs.len().min(goal_obs.len()));
                    let norm_goal = norm_goals[g];
                    let denom = norm_obs * norm_goal;
                    if denom > 0.0 && dot / denom > 0.9 {
                        mask[g] = true;
                    }
                }
            }
        }
        mask
    }

    /// Number of goals completed in the current episode.
    /// Enables "first return then explore" — after achieving one goal,
    /// immediately sample another. Resets to 0 on episode end.
    fn goals_completed_this_episode(&self) -> usize {
        0
    }

    /// Whether to only sample from previously seen goals.
    /// From JAX `ONLY_SAMPLE_FROM_SEEN_GOALS` config flag.
    fn only_sample_from_seen(&self) -> bool {
        true
    }
}

// ── SpeculativeGenerator (Plan 193 Phase 1) ──────────────────

/// Generic trait for speculative generation of candidates.
///
/// Decouples the output format (tokens, actions, etc.) from the generation
/// mechanism. Any domain implementing `generate() + validate()` can reuse
/// the same DDTree + pruning + routing infrastructure.
///
/// This is the Generator Contract pattern: open for extension (new domains),
/// closed for modification.
pub trait SpeculativeGenerator {
    /// Input condition type (e.g., token context, game state).
    type Condition;
    /// Output type (e.g., logit vector, game action).
    type Output;
    /// Error type.
    type Error;

    /// Generate candidate outputs given a condition.
    fn generate(
        &mut self,
        condition: &Self::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error>;

    /// Batch variant for GPU amortization.
    #[inline]
    fn generate_batch(
        &mut self,
        conditions: &[Self::Condition],
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Vec<Self::Output>>, Self::Error> {
        conditions.iter().map(|c| self.generate(c, rng)).collect()
    }
}

/// Typed constraint pruner for generic outputs.
///
/// Extends ConstraintPruner concept to typed output validation.
pub trait GenerativeConstraintPruner<Output>: Send + Sync {
    /// Returns true if the output passes all constraints.
    fn is_valid(&self, output: &Output) -> bool;

    /// Batch variant for amortization.
    #[inline]
    fn batch_is_valid(&self, outputs: &[Output]) -> Vec<bool> {
        outputs.iter().map(|o| self.is_valid(o)).collect()
    }
}

// ── RecursionLogits (Plan 283 T2.3) ──────────────────────────────

/// Opt-in trait for generators that expose pre/post recursion logits.
///
/// Distilled from [arxiv:2511.16886](https://arxiv.org/abs/2511.16886).
/// See `.plans/283_self_advantage_recursion_gate.md`, `.research/250_*.md`.
///
/// Generators that perform iterative recursion (e.g., weight-shared loops,
/// multi-step refine) implement this trait so that an `AdvantageMarginGate`
/// can detect dead-compute steps and halt recursion early.
///
/// # Why a separate trait (not extending `SpeculativeGenerator`)
///
/// Modifying `SpeculativeGenerator` would force every implementor to expose
/// pre-recursion logits, even generators that have no recursion concept
/// (e.g., single-pass game-action generators). This opt-in trait avoids
/// trait breakage: only generators with a recursion loop implement it.
///
/// # Integration pattern
///
/// ```text
/// pub trait SpeculativeGenerator { ... }            // unchanged
/// pub trait RecursionLogits { ... }                 // opt-in
///
/// impl<G: SpeculativeGenerator + RecursionLogits> RecursionGatedGenerator<G> { ... }
/// ```
#[cfg(feature = "recursion_logits")]
pub trait RecursionLogits {
    /// Pre-recursion logits `π̂` from the most recent generate() call.
    ///
    /// Returns the logits snapshot captured BEFORE the recursion loop ran.
    /// Empty slice if no recursion occurred (single-pass generators may
    /// return `&[]`).
    fn pre_recursion_logits(&self) -> &[f32];

    /// Post-recursion logits `π+` from the most recent generate() call.
    ///
    /// Returns the logits snapshot captured AFTER the recursion loop
    /// completed (or was halted by a gate). Same length as
    /// `pre_recursion_logits()` when both are non-empty.
    fn post_recursion_logits(&self) -> &[f32];
}

// ── QGradientOracle (Plan 268 — QGF Test-Time Q-Guided Flow) ─────

/// Critic gradient oracle for test-time guidance.
///
/// Provides `∇_a Q(s, a)` — the gradient of the critic (value function)
/// with respect to the action — evaluated at a *projected* final output.
/// This is the modelless primitive that powers QGF (Q-Guided Flow),
/// distilling the test-time gradient-guidance principle from
/// arXiv:2606.11087 (Zhou et al., 2026).
///
/// # QGF Design Decision: Drop the Jacobian (J ≈ I)
///
/// Per the QGF paper §5, the gradient is computed at a **first-order
/// Euler approximation** of the clean output (`â_1 = a_t + (1-t)·v_θ`),
/// and the Jacobian `∂â_1/∂a_t` is intentionally **dropped** (set to
/// identity). This is *not* a lazy approximation:
///
/// - It gives **lower variance** than the full BPTT gradient (paper Fig 3)
/// - It is **cheaper** (no backprop through the generator)
/// - It produces **better** Q-optimization (paper Fig 4)
///
/// **Do NOT add chain-rule backprop through the generator** in downstream
/// implementations — it re-introduces the high-variance BPTT path that QGF
/// was designed to avoid. The existing FFT smoothing in `FlowFieldCache`
/// is the equivalent variance-reduction mechanism for our discrete case.
///
/// # Usage
///
/// Implement this for any value-function-like type:
/// - `LeoHead` (Q-values for all goals/actions)
/// - `FlowFieldCache` (Q-values → FFT smoothed → gradient field)
/// - `ActionBridge` (latent → raw action via ternary direction dot-product)
/// - A BFN-rejection-sampling proxy (Freeze-tier fallback, no trained critic)
///
/// See `.research/236_QGF_Test_Time_Q_Guided_Flow.md` and
/// `.plans/268_qgf_test_time_q_guided_flow.md`.
#[cfg(feature = "qgf_oracle")]
pub trait QGradientOracle {
    /// State type (observation, context).
    type State;

    /// Action type (token, game action, latent vector).
    type Action;

    /// `∇_a Q(s, a)` evaluated at the *projected* final action.
    ///
    /// The caller must first produce the projection `â_1` via
    /// `qgf::project_one_step()` — querying the gradient at the *current*
    /// intermediate action is the OOD-biased path that QGF avoids.
    ///
    /// Returns a vector of per-action-dimension gradients. For discrete
    /// action spaces, this is a per-action logit tilt.
    fn q_gradient_at(&self, state: &Self::State, projected_action: &Self::Action) -> Vec<f32>;

    /// Zero-alloc variant — writes the gradient into the caller-provided buffer.
    ///
    /// `out.len()` must equal the action-space dimensionality. Implementations
    /// should panic (debug) or no-op (release) on length mismatch.
    fn q_gradient_into(
        &self,
        state: &Self::State,
        projected_action: &Self::Action,
        out: &mut [f32],
    );

    /// Confidence in the gradient at this state, in `[0.0, 1.0]`.
    ///
    /// Used by `VarianceAdaptiveGuidance` (Plan 268 F4) to scale the guidance
    /// weight `1/β` per-query via `sigmoid(k · (confidence − threshold))`.
    ///
    /// - Returns `1.0` for deterministic oracles (cached Q-values, ternary bridge)
    /// - Returns lower values for noisy oracles (BFN rejection proxy, stale critic)
    ///
    /// Reuse Thicket (Plan 267) variance probe if available — confidence is
    /// `1.0 − normalized_variance(Q(s, ·))`.
    #[inline]
    fn confidence(&self, _state: &Self::State) -> f32 {
        1.0
    }
}

/// No-op oracle that returns zero gradient — pure BC reference policy,
/// no Q-guidance. Used in the Freeze tier (graceful degradation when no
/// trained critic is available). Engine always boots.
#[cfg(feature = "qgf_oracle")]
#[derive(Clone, Debug, Default)]
pub struct NoGuidanceOracle;

#[cfg(feature = "qgf_oracle")]
impl QGradientOracle for NoGuidanceOracle {
    type State = ();
    type Action = ();

    #[inline]
    fn q_gradient_at(&self, _state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
        Vec::new()
    }

    #[inline]
    fn q_gradient_into(
        &self,
        _state: &Self::State,
        _projected_action: &Self::Action,
        out: &mut [f32],
    ) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
    }

    /// Freeze tier: zero confidence → adaptive guidance weight collapses to 0.
    #[inline]
    fn confidence(&self, _state: &Self::State) -> f32 {
        0.0
    }
}

// ── Game Trace + Partial Scoring (Plan 191) ──────────────────────

/// Minimal game trace for partial scoring.
///
/// Captures episode statistics needed for graduated reward computation.
/// Lightweight: stack-friendly, no heap allocations in the struct itself.
#[cfg(feature = "partial_scoring")]
#[derive(Clone, Copy, Debug, Default)]
pub struct GameTrace {
    /// Final reward from the game engine (binary: win=1.0, loss=0.0).
    pub final_reward: f64,
    /// Ticks survived before termination.
    pub survival_ticks: u32,
    /// Opponents eliminated.
    pub kills: u32,
    /// Total actions taken during the episode.
    pub actions_taken: u32,
    /// Maximum possible ticks (episode budget).
    pub max_ticks: u32,
}

/// Graduated reward scorer for game episodes.
///
/// Replaces binary win/loss with continuous [0.0, 1.0] reward,
/// giving bandit algorithms richer signal for faster convergence.
///
/// Ref: FrontierCS (human 95 vs best model 29 on open-ended problems).
#[cfg(feature = "partial_scoring")]
pub trait PartialScorer: Send + Sync {
    /// Compute graduated score from game trace.
    fn partial_score(&self, trace: &GameTrace) -> f32;

    /// Breakdown of per-criteria scores.
    fn score_breakdown(&self, trace: &GameTrace) -> Vec<(&'static str, f32)> {
        vec![("total", self.partial_score(trace))]
    }
}

// ── Game Config + Problem Mutation (Plan 191) ────────────────────

/// Generic game configuration for mutation.
///
/// Represents the tunable parameters of a game arena.
/// Each field can be perturbed by a [`ProblemMutator`] to create
/// harder variants for open-ended problem evolution.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    /// Maximum steps per episode.
    pub max_steps: u32,
    /// Grid/board size (e.g., 9 for 9x9, 15 for 15x15).
    pub grid_size: u32,
    /// Number of opponents/NPCs.
    pub opponent_count: u32,
    /// Weight for survival objective in scoring.
    pub survival_weight: f32,
    /// Weight for kill/objective in scoring.
    pub kill_weight: f32,
}

#[cfg(feature = "problem_mutator")]
impl Default for GameConfig {
    fn default() -> Self {
        Self {
            max_steps: 200,
            grid_size: 9,
            opponent_count: 1,
            survival_weight: 0.5,
            kill_weight: 0.5,
        }
    }
}

/// Mutation kind for game configs.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MutationKind {
    /// Shift objective weights (e.g., survival vs kills).
    GoalReweight,
    /// Reduce action space or add constraints.
    ConstrainOutputs,
    /// Vary input parameters (grid size, opponent count).
    GeneralizeInputs,
}

/// A mutated game config with estimated difficulty delta.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Debug)]
pub struct MutantConfig {
    /// Which mutation strategy was applied.
    pub mutation_kind: MutationKind,
    /// Human-readable description of the mutation.
    pub description: String,
    /// Estimated difficulty increase over seed config.
    pub difficulty_delta: f32,
}

/// Trait for mutating game configs into harder variants.
///
/// Implements the FrontierSmith closed→open problem synthesis:
/// take a base game config and produce progressively harder variants.
/// The arena scheduler feeds these mutated configs to rounds.
#[cfg(feature = "problem_mutator")]
pub trait ProblemMutator: Send + Sync {
    /// Mutate a seed config into variants.
    fn mutate(&self, seed: &GameConfig) -> Vec<MutantConfig>;
}

// ── Best Buddies Drafting (Plan 199) ────────────────────────────

/// Pearson correlation coefficient between two f32 slices.
///
/// SIMD-friendly: single pass, no allocation.
/// Returns 0.0 if either slice has zero variance.
#[inline]
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    // Single-pass Welford-style: track sum_a, sum_b, cov, var_a, var_b.
    // Uses corrective term (n·Σxy − Σx·Σy) to match two-pass numerically.
    //
    // SIMD: the four Σ reductions run through crate::simd helpers (NEON/AVX2);
    // only the final corrective arithmetic stays in f64 for numerical parity
    // with the original Welford-style two-pass reference.
    let len = a.len();
    let sum_a = crate::simd::simd_sum_f32(a) as f64;
    let sum_b = crate::simd::simd_sum_f32(b) as f64;
    let sum_ab = crate::simd::simd_dot_f32(a, b, len) as f64;
    let sum_aa = crate::simd::simd_dot_f32(a, a, len) as f64;
    let sum_bb = crate::simd::simd_dot_f32(b, b, len) as f64;
    let n = len as f64;
    let cov = n * sum_ab - sum_a * sum_b;
    let var_a = n * sum_aa - sum_a * sum_a;
    let var_b = n * sum_bb - sum_b * sum_b;
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (cov / denom) as f32
}

/// Best buddies: mutual nearest neighbors from correlation rows.
///
/// Returns pairs `(i, j)` where row `i`'s best match is row `j` AND row `j`'s
/// best match is row `i`. Results sorted by correlation magnitude (descending),
/// truncated to top-`k`.
pub fn best_buddies(corr_rows: &[&[f32]], k: usize) -> Vec<(usize, usize)> {
    let n = corr_rows.len();

    // Build best-match index for each row
    let mut best_for: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let row = corr_rows[i];
        if row.is_empty() {
            continue;
        }
        // Single-pass SIMD argmax: fuses max-finding + index-recovery into one
        // traversal. NEON kernel is ~5× faster than the prior two-pass idiom
        // (simd_max_f32 + position scan). Tie-break matches first-wins semantics.
        let (best_j, best_corr) = crate::simd::simd_argmax_f32(row);
        let _ = best_corr; // value unused; only index is needed below
        best_for[i] = Some(best_j);
    }

    // Mutual agreement: i→j AND j→i
    let mut buddies: Vec<(usize, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let Some(j) = best_for[i] else { continue };
        // j must be within bounds and point back to i
        if j >= n {
            continue;
        }
        match best_for[j] {
            Some(k_idx)
                if k_idx == i
                // Avoid double-counting: only push when i < j
                && i < j =>
            {
                buddies.push((i, j));
            }
            _ => {}
        }
    }

    // Sort by correlation magnitude (descending), keep top-k.
    // Unstable is safe: only top-K is consumed; tie order within K is unspecified
    // (the doc contract is "top-K by magnitude", not a stable ordering).
    buddies.sort_unstable_by(|a, b| {
        let corr_a = corr_rows[a.0][a.1];
        let corr_b = corr_rows[b.0][b.1];
        corr_b.total_cmp(&corr_a)
    });
    buddies.truncate(k);
    buddies
}

/// Best Buddies filter for speculative decoding.
///
/// Mines mutual agreement between draft and target model marginals.
/// Zero training — purely inference-time correlation.
pub trait BestBuddyAligner: Send + Sync {
    /// Compute mutual agreement score for a token position.
    ///
    /// Returns 1.0 if token is a best buddy (draft prefers it AND target prefers it),
    /// 0.0 otherwise. Continuous in between via EMA correlation.
    fn mutual_agreement(&self, draft_top_k: &[f32], target_top_k: &[f32]) -> f32;

    /// Batch version: compute alignment confidence for all positions.
    ///
    /// `draft_logits` and `target_logits` are `[seq_len × vocab_size]` (flat),
    /// `results` receives `[seq_len]` confidence scores.
    fn batch_alignment_confidence(
        &self,
        draft_logits: &[f32],
        target_logits: &[f32],
        results: &mut [f32],
    );
}

// ── LEO Tests (Plan 155, T7) ────────────────────────────────────

// Tests (extracted to sibling files — see Issue 174)
mod tests_leo;

#[cfg(test)]
mod tests_spec_gen;

#[cfg(test)]
mod tests_action_space_log; // Issue 690: f64 accumulator regression gates

#[cfg(test)]
mod tests_best_buddies;

#[cfg(test)]
mod tests_reject_confidence;

#[cfg(all(test, feature = "recursion_logits"))]
mod recursion_logits_tests;
