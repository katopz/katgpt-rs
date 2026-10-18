//! Structural CoT halting — answer-space cycle detection on reasoning traces
//! (Issue 699 T1–T3; Research 525; arXiv:2510.07880 "TRACE").
//!
//! Distilled from TRACE (Google DeepMind/UMich): structural analysis of LLM
//! overthinking + two ground-truth-free real-time halting heuristics. This is
//! a NEW signal class beside the numeric halters (entropy, residual,
//! gain/cost, patience-on-score): halting on the trace's own **answer-space
//! structure**, which works black-box — no logits, no hidden states, no LLM
//! rater, no training. Modelless by construction (fixed arrays + BLAKE3
//! hashing + plain comparisons; zero RNG, zero softmax, zero new deps).
//!
//! # Mechanism
//!
//! [`StructuralTraceMonitor`] consumes a stream of answer-bearing reasoning
//! steps (one normalized answer per step) and classifies each step's
//! answer-space transition (the truth table is documented on
//! [`StructuralTransition`] and pinned by tests):
//!
//! - `Verify` — the answer is unchanged (the model is re-deriving / verifying
//!   the same answer).
//! - `Correct` — the answer changed to something never seen (or ring-evicted);
//!   the previous answer is marked abandoned.
//! - `BacktrackRevisit` — the answer changed BACK to a previously-abandoned
//!   answer still in the ring → an answer-space cycle (the model is going in
//!   circles, not making progress).
//!
//! Two halting policies consume those events ([`SelfLoopHalt`], the paper's
//! K-consecutive-verifications rule with paper default K=2;
//! [`BacktrackRevisitHalt`], halt on the first revisit-on-backtrack), plus
//! [`HaltPolicy::Auto`] — the pattern-conditional fusion (T3): the monitor
//! classifies the trace prefix as [`Pattern::Explorer`] or
//! [`Pattern::LateLanding`] from the answer histogram's collision purity
//! (the shipped `crate::ict::math::collision_purity`, Plan 294 — NOT
//! reimplemented here) + the positional mass of answer changes, then selects
//! the policy the paper hand-tuned per model family — derived here
//! deterministically (see [`StructuralTraceMonitor::classify_prefix`]).
//!
//! The decisions compose as a THIRD independent halt-vote family beside the
//! hidden-state residual (FPRM, Plan 266) and gain/cost (Plans 282/304) via
//! [`compose_votes`] / [`vote_from_numeric`] — mirroring the
//! `GainCostLoopHalter::halt_decision` arbiter ergonomics so a consumer can
//! run both families and merge.
//!
//! # Transition predicate (truth table, pinned by `transition_truth_table`)
//!
//! State: `current` = the previous step's normalized answer identity (`None`
//! before the first step). The answer ring stores one entry per distinct
//! answer with an `abandoned` flag (set when the answer is LEFT, cleared when
//! it becomes current again) and an activation step.
//!
//! | current | new answer | in ring? | abandoned since activation? | transition |
//! |---|---|---|---|---|
//! | None | A | – | – | `Correct` (establishment; NOT a positional change) |
//! | A | A | – | – | `Verify` (`verify_run += 1`) |
//! | A | B ∉ ring | no | – | `Correct` (push ring; A marked abandoned; a positional change) |
//! | A | B ∈ ring | yes | yes | `BacktrackRevisit` — answer-space cycle |
//! | A | B ∈ ring | yes | no | `Correct` (defensive re-activation; structurally unreachable — a non-current entry was necessarily left at least once) |
//!
//! The "abandoned since activation" flag IS the issue text's "≥1 abandonment
//! has occurred since that entry was pushed": entries are re-activated (flag
//! cleared, activation step refreshed) on revisit, so a second A→B→A cycle
//! re-arms the flag and fires the policy again (pinned by
//! `revisit_refires_after_re_abandonment`).
//!
//! # Ring wraparound (bounded-memory semantics)
//!
//! The ring holds [`RING_CAPACITY`] = 8 distinct answers; inserting a 9th
//! evicts the OLDEST slot (circular write pointer). An evicted answer later
//! re-proposed classifies `Correct` (fresh) — a cycle longer than 8 distinct
//! answers is not detected. That is the documented bound; the paper's cycles
//! are short (2–3 answers). Hash collisions (64-bit BLAKE3 truncation) would
//! conflate two answers as one — negligible at ring scale.
//!
//! # Episode semantics
//!
//! One monitor = one trace episode. The FIRST `Halt` is recorded and every
//! subsequent `step` returns that same decision without consuming state (a
//! halt ends the measurement episode — same shape as the numeric arbiter's
//! episode discipline). [`StructuralTraceMonitor::reset`] starts a fresh
//! episode with the same policy.
//!
//! # Latent vs Raw
//!
//! The halt decision (`reason: u8`, `step: usize`) is a deterministic raw
//! scalar — safe to sync/replay. The answer ring/hash state is LOCAL
//! trace-derived text identity: never synced raw; only the scalar decision
//! crosses any boundary.
//!
//! # Zero-alloc contract
//!
//! `step` / `step_key` / `classify_prefix` allocate nothing: normalization
//! streams over `&str` bytes into a stack BLAKE3 hasher (the answer String is
//! never materialized), and every table is a fixed `[…; 8]` array. Pinned by
//! the integration target's G4 audit (10k+ steps, counting allocator).
//!
//! Issue 699 T1-T3 — structural CoT halting (TRACE, arXiv:2510.07880):
//! answer-space cycle detection on reasoning traces — the black-box halt
//! family (no logits/hidden states/LLM rater). A third independent halt-vote
//! family beside the numeric arbiter above; the two compose via
//! structural_cot_halt::vote_from_numeric when BOTH features are on.
//! Opt-in (not default-on) — T4 defend-wrong PoC (riir-poc) + T5 GOAT gate
//! (≥30% token savings at ≤1% accuracy delta) are pending.

#![allow(clippy::float_cmp)] // float comparisons in tests against exact constants

use crate::ict::math::collision_purity;

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/// Distinct-answer capacity of the ring and the answer histogram.
///
/// 8 covers the paper's observed cycle lengths (2–3 answers) with headroom;
/// wraparound evicts the oldest entry (documented limitation: cycles longer
/// than 8 distinct answers are not detected).
pub const RING_CAPACITY: usize = 8;

/// Paper default self-loop patience: halt after `K = 2` consecutive
/// self-verifications of the current answer (arXiv:2510.07880 §heuristics —
/// K=2 halves output length at 62.23/68.90 acc on their Qwen3/R1 traces).
pub const SELF_LOOP_DEFAULT_K: u32 = 2;

/// Late-Landing self-loop patience: the paper's per-model tuning gives
/// K=3 for the Late-Landing family (Qwen3-32B: 80.18 acc at −40% cost).
/// Our fusion derives K deterministically instead of hand-tuning it:
/// K=3 when the landing answer is still contested, K=2 when it dominates —
/// see [`LATE_LANDING_PURITY_K2`].
pub const LATE_LANDING_DEFAULT_K: u32 = 3;

/// Purity grade at which a classified Late-Landing trace drops from
/// K=[`LATE_LANDING_DEFAULT_K`] to K=2.
///
/// `collision_purity >= 0.75` means the landing answer holds ≥ ~87% of all
/// steps (π ≥ √0.75 ≈ 0.866) — the trace is deep in verification of a
/// dominant answer, so the paper's cheaper general default (K=2) applies.
/// Below that the landing answer is still contested and K=3 buys one more
/// verification before cutting. A plain `>=` comparison — no softmax, no
/// RNG, no learned parameters.
pub const LATE_LANDING_PURITY_K2: f32 = 0.75;

// ─────────────────────────────────────────────────────────────────────
// Answer normalization (text variant)
// ─────────────────────────────────────────────────────────────────────

/// Normalize an answer and hash it to a 64-bit identity.
///
/// Normalization: trim leading/trailing ASCII whitespace, ASCII-lowercase
/// (A–Z only; Unicode case folding is out of scope for this PoC and
/// documented), collapse internal ASCII-whitespace runs to a single space.
/// The normalized byte stream is fed to BLAKE3 incrementally — the
/// normalized `String` is NEVER materialized — and the first 8 digest bytes
/// are taken little-endian.
///
/// The empty answer is a valid answer (a model emitting nothing): it hashes
/// to BLAKE3 of the empty stream, distinct from every non-empty answer.
///
/// Deterministic across platforms/runs (BLAKE3 + byte-wise ASCII ops; no
/// locale, no RNG, no HashMap iteration).
#[inline]
pub fn normalized_answer_hash(answer: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    let bytes = answer.as_bytes();
    // Trim leading whitespace.
    let mut start = 0;
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    // Trim trailing whitespace.
    let mut end = bytes.len();
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    // Stream the trimmed body: ASCII-lowercase each byte, collapse interior
    // whitespace runs to one space. Zero allocations — the hasher is a stack
    // value and `update` consumes borrowed bytes.
    let mut in_ws = false;
    let mut i = start;
    while i < end {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            if !in_ws {
                hasher.update(b" ");
                in_ws = true;
            }
        } else {
            hasher.update(&[b.to_ascii_lowercase()]);
            in_ws = false;
        }
        i += 1;
    }
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().expect("BLAKE3 digest is 32 bytes"))
}

// ─────────────────────────────────────────────────────────────────────
// Decision types (mirroring gain_cost_halt's ergonomics)
// ─────────────────────────────────────────────────────────────────────

/// The answer-space event class of one reasoning step.
///
/// See the module-level truth table for the exact predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StructuralTransition {
    /// Answer unchanged from the previous step (`verify_run += 1`).
    Verify,
    /// Answer changed to a fresh answer (ring push) — or the first step
    /// establishing the initial answer (which is NOT a positional change).
    Correct,
    /// Answer changed BACK to a previously-abandoned ring entry — an
    /// answer-space cycle.
    BacktrackRevisit,
}

/// Why a structural halt fired (or which family a composed vote came from).
///
/// `#[repr(u8)]` keeps the payload at 1 byte so [`StructuralHaltDecision`]
/// stays cache-word sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StructuralHaltReason {
    /// `SelfLoopHalt` fired: `k` consecutive `Verify` transitions of the
    /// current answer (the paper's self-looping rule).
    SelfLoop = 0,
    /// `BacktrackRevisitHalt` fired: the model revisited a previously
    /// abandoned answer (the paper's backtrack rule — an answer-space cycle).
    BacktrackRevisit = 1,
    /// A vote recorded on behalf of an INDEPENDENT (e.g. numeric) halt
    /// family through [`vote_from_numeric`] — the structural vote surface is
    /// the composition point, so foreign decisions ride this variant. The
    /// monitor itself never produces it.
    NumericArbiter = 2,
}

/// Result of one structural trace evaluation.
///
/// Returned by [`StructuralTraceMonitor::step`] /
/// [`StructuralTraceMonitor::step_key`] each step. The caller maps this onto
/// its loop-control flow:
/// - [`StructuralHaltDecision::Continue`] → keep generating.
/// - [`StructuralHaltDecision::Halt`] → stop; the `step` field is the 1-based
///   index of the step that triggered the halt.
///
/// Mirrors `gain_cost_halt::HaltDecision` (minus its `RefusedFloor` arm —
/// answer-space structure has no representational floor; use
/// [`HaltPolicy::Never`] for a guaranteed no-op control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StructuralHaltDecision {
    /// Keep going — no policy fired on this step.
    Continue,
    /// Halt now — a policy fired on this step.
    Halt {
        /// Which policy family fired.
        reason: StructuralHaltReason,
        /// 1-based index of the step that triggered the halt.
        step: usize,
    },
}

impl StructuralHaltDecision {
    /// `true` iff this is a [`StructuralHaltDecision::Halt`].
    #[inline]
    pub const fn is_halt(&self) -> bool {
        matches!(self, StructuralHaltDecision::Halt { .. })
    }
}

/// The halt-vote type: one family's opinion for one step.
///
/// A type alias (not a newtype) so votes from both families flow through the
/// same [`compose_votes`] surface without conversion ceremony. The structural
/// monitor produces its votes from `step`/`halt_vote`; numeric-family votes
/// enter via [`vote_from_numeric`].
pub type HaltVote = StructuralHaltDecision;

// ─────────────────────────────────────────────────────────────────────
// Policies
// ─────────────────────────────────────────────────────────────────────

/// The paper's self-looping policy: halt after `k` consecutive
/// self-verifications of the current answer.
///
/// Semantics (pinned by `self_loop_fires_at_exactly_k`): `verify_run` counts
/// consecutive `Verify` transitions, and the halt fires on the step where
/// `verify_run >= k`. With the paper default K=2 the minimum trace is
/// `[A, A, A]` — one answer proposal followed by K=2 verification steps —
/// and the halt lands on the K-th verification (step 3), never earlier
/// (the (K−1)-th verification continues).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfLoopHalt {
    /// Consecutive verifies required before halting. `k = 0` behaves as
    /// `k = 1` (the first verify carries `verify_run = 1`).
    pub k: u32,
}

impl SelfLoopHalt {
    /// The paper's general default (K=2).
    #[inline]
    pub const fn paper_default() -> Self {
        Self {
            k: SELF_LOOP_DEFAULT_K,
        }
    }

    /// Construct with an explicit patience, clamped to ≥ 1.
    #[inline]
    pub fn new(k: u32) -> Self {
        Self { k: k.max(1) }
    }
}

impl Default for SelfLoopHalt {
    /// [`SelfLoopHalt::paper_default`] (K=2).
    #[inline]
    fn default() -> Self {
        Self::paper_default()
    }
}

impl From<SelfLoopHalt> for HaltPolicy {
    #[inline]
    fn from(s: SelfLoopHalt) -> Self {
        HaltPolicy::SelfLoop(s.k)
    }
}

/// The paper's backtrack policy: halt the first time the model revisits a
/// previously-abandoned answer (a [`StructuralTransition::BacktrackRevisit`]
/// transition) — the answer-space cycle signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BacktrackRevisitHalt;

impl From<BacktrackRevisitHalt> for HaltPolicy {
    #[inline]
    fn from(_: BacktrackRevisitHalt) -> Self {
        HaltPolicy::BacktrackRevisit
    }
}

/// Which halting policy the monitor evaluates each step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HaltPolicy {
    /// Pattern-conditional fusion (T3): re-classify the trace prefix each
    /// step (cheap — O(ring) over fixed arrays) and evaluate the step with
    /// the pattern's selected policy. Explorer → [`HaltPolicy::BacktrackRevisit`];
    /// Late Landing → [`HaltPolicy::SelfLoop`] with the purity-graded K
    /// (see [`StructuralTraceMonitor::classify_prefix`]). Deterministic: the
    /// classification is a pure function of the consumed prefix.
    Auto,
    /// [`SelfLoopHalt`] with the given patience.
    SelfLoop(u32),
    /// [`BacktrackRevisitHalt`].
    BacktrackRevisit,
    /// Never halt — the no-op control. This is the behavioral shape of the
    /// feature being OFF (the whole module is `structural_cot_halt`-gated;
    /// with the flag off the monitor simply does not exist), and it doubles
    /// as the composition-safety control arm for the T4 PoC.
    Never,
}

impl Default for HaltPolicy {
    /// [`HaltPolicy::Auto`] — the fusion is the paper's recommended
    /// configuration (its two heuristics are per-pattern, not global).
    #[inline]
    fn default() -> Self {
        HaltPolicy::Auto
    }
}

// ─────────────────────────────────────────────────────────────────────
// T3 pattern classifier
// ─────────────────────────────────────────────────────────────────────

/// The two trace patterns the paper's halting heuristics are conditioned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Pattern {
    /// The model wanders: answers keep changing, verifications never
    /// accumulate, cycles are common. Paper exemplar: Qwen3-235B.
    /// Policy: backtrack-revisit (preserves accuracy at ~60% savings).
    Explorer,
    /// The model lands on its answer late and then verifies it repeatedly.
    /// Paper exemplar: Qwen3-32B. Policy: self-loop K (paper: 3).
    LateLanding,
}

/// A classification verdict: the pattern AND the policy it selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassifiedPattern {
    /// Which pattern the consumed prefix matches.
    pub pattern: Pattern,
    /// The policy that pattern selects (never [`HaltPolicy::Auto`] — the
    /// fusion resolves Auto to a concrete policy).
    pub policy: HaltPolicy,
}

// ─────────────────────────────────────────────────────────────────────
// Monitor
// ─────────────────────────────────────────────────────────────────────

/// Structural CoT halt monitor — the Issue 699 T1 core.
///
/// Consumes a stream of answer-bearing reasoning steps (one normalized
/// answer per step, via [`step`](StructuralTraceMonitor::step)) or raw
/// 64-bit answer identities (via
/// [`step_key`](StructuralTraceMonitor::step_key) — the token-space seam
/// used by the dd-tree/MCTS plug points, where "answers" are token ids and
/// hashing is skipped by passing the id verbatim). Both entries share one
/// state machine; never mix text and raw keys in one monitor.
///
/// Fixed-size state (~300 B, no heap): asserted ≤ 512 B by
/// `size_bounds_stay_cache_friendly`.
#[derive(Clone, Debug)]
pub struct StructuralTraceMonitor {
    /// Active policy (or [`HaltPolicy::Auto`] — resolved per step).
    policy: HaltPolicy,
    /// Current answer identity (`None` before the first step).
    current_key: Option<u64>,
    /// Ring slot of the current answer (valid iff `current_key.is_some()`).
    current_slot: u8,
    /// Consecutive `Verify` transitions of the current answer.
    verify_run: u32,
    /// Total `BacktrackRevisit` events observed (the issue's §5.1 bonus:
    /// a revisited answer ≈ two independent derivations — usable as a
    /// credibility/vote weight by T6 consumers).
    revisit_count: u32,
    /// 1-based count of steps consumed.
    step_count: usize,
    /// 1-based step index of the last positional change (Correct-with-change
    /// or BacktrackRevisit). `0` = no change yet (a single-answer trace).
    last_change_step: usize,
    // ── Answer ring (insertion order, circular write pointer) ──
    ring_hash: [u64; RING_CAPACITY],
    ring_activation_step: [u32; RING_CAPACITY],
    ring_abandoned: [u8; RING_CAPACITY],
    /// Next ring write slot (wraparound evicts the oldest).
    ring_next: u8,
    /// Live ring entries (≤ [`RING_CAPACITY`]).
    ring_len: u8,
    // ── Answer histogram (for the T3 classifier) ──
    hist_hash: [u64; RING_CAPACITY],
    hist_count: [u32; RING_CAPACITY],
    hist_len: u8,
    // ── Episode state ──
    halted: bool,
    halt_decision: StructuralHaltDecision,
    last_transition: Option<StructuralTransition>,
}

impl StructuralTraceMonitor {
    /// Construct a monitor with an explicit policy.
    #[inline]
    pub fn new(policy: HaltPolicy) -> Self {
        Self {
            policy,
            current_key: None,
            current_slot: 0,
            verify_run: 0,
            revisit_count: 0,
            step_count: 0,
            last_change_step: 0,
            ring_hash: [0; RING_CAPACITY],
            ring_activation_step: [0; RING_CAPACITY],
            ring_abandoned: [0; RING_CAPACITY],
            ring_next: 0,
            ring_len: 0,
            hist_hash: [0; RING_CAPACITY],
            hist_count: [0; RING_CAPACITY],
            hist_len: 0,
            halted: false,
            halt_decision: StructuralHaltDecision::Continue,
            last_transition: None,
        }
    }

    /// Construct with [`HaltPolicy::Auto`] — the pattern-conditional fusion.
    #[inline]
    pub fn auto() -> Self {
        Self::new(HaltPolicy::Auto)
    }

    /// Consume one answer-bearing reasoning step (text variant).
    ///
    /// Normalizes + hashes the answer (zero-alloc — see
    /// [`normalized_answer_hash`]) and advances the state machine. Returns
    /// the halt decision for this step; after the first
    /// [`StructuralHaltDecision::Halt`] the episode is frozen and every
    /// further call returns the recorded decision unchanged.
    #[inline]
    pub fn step(&mut self, answer: &str) -> StructuralHaltDecision {
        self.step_key(normalized_answer_hash(answer))
    }

    /// Composition-surface alias for [`Self::step`] — the monitor's halt
    /// vote for this step, feedable to [`compose_votes`].
    #[inline]
    pub fn halt_vote(&mut self, answer: &str) -> HaltVote {
        self.step(answer)
    }

    /// Raw-identity variant of [`Self::step`] — the token-space seam for
    /// consumers whose "answers" are already integers (dd-tree dominant
    /// tokens, MCTS root actions). The key is used verbatim as the answer
    /// identity (no hashing, no allocation). Do NOT mix raw keys with text
    /// hashes in one monitor.
    pub fn step_key(&mut self, key: u64) -> StructuralHaltDecision {
        // Episode freeze: a halt ends the measurement episode.
        if self.halted {
            return self.halt_decision;
        }
        let transition = self.observe_key(key);
        self.last_transition = Some(transition);
        let step = self.step_count;

        // Resolve the active policy (the fusion re-classifies the prefix
        // each step — O(2 × ring) over fixed arrays).
        let active_policy = match self.policy {
            HaltPolicy::Auto => self.classify_prefix().policy,
            p => p,
        };

        let decision = match (active_policy, transition) {
            (HaltPolicy::SelfLoop(k), StructuralTransition::Verify) if self.verify_run >= k => {
                StructuralHaltDecision::Halt {
                    reason: StructuralHaltReason::SelfLoop,
                    step,
                }
            }
            (HaltPolicy::BacktrackRevisit, StructuralTransition::BacktrackRevisit) => {
                StructuralHaltDecision::Halt {
                    reason: StructuralHaltReason::BacktrackRevisit,
                    step,
                }
            }
            _ => StructuralHaltDecision::Continue,
        };

        match decision {
            StructuralHaltDecision::Halt { .. } => {
                self.halted = true;
                self.halt_decision = decision;
            }
            StructuralHaltDecision::Continue => {}
        }
        decision
    }

    /// Raw-identity composition alias for [`Self::step_key`].
    #[inline]
    pub fn halt_vote_key(&mut self, key: u64) -> HaltVote {
        self.step_key(key)
    }

    /// Classify the consumed prefix (T3 pattern-conditional fusion).
    ///
    /// Cheap and re-runnable as the trace grows: O(2 × ring) over fixed
    /// arrays, zero allocation, no mutation. Deterministic by construction.
    ///
    /// # The deterministic derivation (replacing the paper's per-model
    /// hand-tuning)
    ///
    /// Two signals, both mandated by the issue:
    ///
    /// 1. **Positional mass of answer changes** — the verify-tail fraction.
    ///    `steps_since_last_change × 2 >= step_count` (integer comparison;
    ///    a trace with no change at all trivially qualifies). At least half
    ///    the trace is pure verification of one answer ⇒ the model has
    ///    landed ⇒ [`Pattern::LateLanding`]. Otherwise changes are still
    ///    arriving ⇒ wandering ⇒ [`Pattern::Explorer`].
    /// 2. **Collision purity of the answer histogram**
    ///    (`collision_purity` over the count-normalized histogram — the
    ///    shipped Plan 294 kernel, NOT reimplemented) — grades the
    ///    Late-Landing self-loop patience: `purity >=
    ///    [`LATE_LANDING_PURITY_K2`] (0.75, i.e. the landing answer holds
    ///    ≥ ~87% of steps) ⇒ K=2 (the paper's cheap general default);
    ///    otherwise K=3 (the paper's Late-Landing value — one more
    ///    verification before cutting a contested landing).
    ///
    /// Purity deliberately does NOT select the pattern: a two-answer
    /// oscillation (A,B,A,B,…) has purity exactly 0.5 — as high as a
    /// converged pair — yet no verify tail. Positional mass separates them
    /// cleanly (pinned by
    /// `classify_two_answer_oscillation_is_explorer_despite_purity_half`).
    /// No softmax anywhere: histograms are count-normalized by division;
    /// thresholds are plain comparisons.
    ///
    /// # Empty trace
    ///
    /// Nothing observed: returns LateLanding + K=2 (the paper's general
    /// default — with no evidence, trust cheaply and cut early).
    pub fn classify_prefix(&self) -> ClassifiedPattern {
        // Collision purity over the count-normalized histogram.
        let mut total = 0u32;
        let hist_len = self.hist_len as usize;
        let mut i = 0;
        while i < hist_len {
            total += self.hist_count[i];
            i += 1;
        }
        if total == 0 {
            return ClassifiedPattern {
                pattern: Pattern::LateLanding,
                policy: HaltPolicy::SelfLoop(SELF_LOOP_DEFAULT_K),
            };
        }
        let inv_total = 1.0f32 / total as f32;
        let mut probs = [0.0f32; RING_CAPACITY];
        let mut i = 0;
        while i < hist_len {
            probs[i] = self.hist_count[i] as f32 * inv_total;
            i += 1;
        }
        let purity = collision_purity(&probs[..hist_len]);

        // Positional mass: verify-tail fraction ≥ 1/2 (integer form).
        let since_change = self.step_count - self.last_change_step;
        let late_tail = since_change * 2 >= self.step_count;

        if late_tail {
            let k = if purity >= LATE_LANDING_PURITY_K2 {
                SELF_LOOP_DEFAULT_K
            } else {
                LATE_LANDING_DEFAULT_K
            };
            ClassifiedPattern {
                pattern: Pattern::LateLanding,
                policy: HaltPolicy::SelfLoop(k),
            }
        } else {
            ClassifiedPattern {
                pattern: Pattern::Explorer,
                policy: HaltPolicy::BacktrackRevisit,
            }
        }
    }

    // ── Observables (for harnesses / the T4 PoC) ──

    /// Consecutive `Verify` transitions of the current answer.
    #[inline]
    pub const fn verify_run(&self) -> u32 {
        self.verify_run
    }

    /// 1-based count of steps consumed (frozen at the halt step after a halt).
    #[inline]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    /// Total answer-space cycles (BacktrackRevisit events) observed.
    #[inline]
    pub const fn revisit_count(&self) -> u32 {
        self.revisit_count
    }

    /// The most recent step's transition (`None` before the first step).
    #[inline]
    pub const fn last_transition(&self) -> Option<StructuralTransition> {
        self.last_transition
    }

    /// Has this episode halted?
    #[inline]
    pub const fn is_halted(&self) -> bool {
        self.halted
    }

    /// The recorded halt decision (`Continue` while not halted).
    #[inline]
    pub const fn halt_decision(&self) -> StructuralHaltDecision {
        self.halt_decision
    }

    /// The active policy (as constructed).
    #[inline]
    pub const fn policy(&self) -> HaltPolicy {
        self.policy
    }

    /// Start a fresh episode with the same policy.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new(self.policy);
    }

    // ── Internals ──

    /// Advance the state machine by one answer identity; classify the
    /// transition per the module truth table.
    fn observe_key(&mut self, key: u64) -> StructuralTransition {
        self.step_count += 1;
        let step = self.step_count;
        self.bump_hist(key);

        match self.current_key {
            // Establishment — the first step is a Correct-class event but
            // deliberately NOT a positional change (nothing was left).
            None => {
                let slot = self.ring_insert(key, step);
                self.current_key = Some(key);
                self.current_slot = slot;
                self.verify_run = 0;
                StructuralTransition::Correct
            }
            Some(current) if key == current => {
                self.verify_run = self.verify_run.saturating_add(1);
                StructuralTransition::Verify
            }
            Some(_) => {
                // The current answer is being LEFT — mark it abandoned.
                let old_slot = self.current_slot as usize;
                self.ring_abandoned[old_slot] = 1;
                self.last_change_step = step;

                match self.ring_find(key) {
                    Some(slot) if self.ring_abandoned[slot as usize] == 1 => {
                        // Backtrack revisit — the answer-space cycle event.
                        let s = slot as usize;
                        self.revisit_count = self.revisit_count.saturating_add(1);
                        self.ring_abandoned[s] = 0;
                        self.ring_activation_step[s] = step as u32;
                        self.current_key = Some(key);
                        self.current_slot = slot;
                        self.verify_run = 0;
                        StructuralTransition::BacktrackRevisit
                    }
                    Some(slot) => {
                        // Defensive re-activation (structurally unreachable:
                        // a non-current entry was necessarily left). Kept so
                        // the literal predicate ("in ring AND not abandoned
                        // ⇒ not a revisit") is exactly what the code says.
                        let s = slot as usize;
                        self.ring_activation_step[s] = step as u32;
                        self.current_key = Some(key);
                        self.current_slot = slot;
                        self.verify_run = 0;
                        StructuralTransition::Correct
                    }
                    None => {
                        // Fresh answer — ring push.
                        let slot = self.ring_insert(key, step);
                        self.current_key = Some(key);
                        self.current_slot = slot;
                        self.verify_run = 0;
                        StructuralTransition::Correct
                    }
                }
            }
        }
    }

    /// Insert a key into the ring at the circular write pointer; when full,
    /// the write evicts the oldest entry (wraparound semantics).
    fn ring_insert(&mut self, key: u64, step: usize) -> u8 {
        let slot = self.ring_next as usize;
        self.ring_hash[slot] = key;
        self.ring_activation_step[slot] = step as u32;
        self.ring_abandoned[slot] = 0;
        self.ring_next = ((slot + 1) % RING_CAPACITY) as u8;
        if (self.ring_len as usize) < RING_CAPACITY {
            self.ring_len += 1;
        }
        slot as u8
    }

    /// Linear scan of the live ring entries (fixed ≤ 8 — deterministic).
    fn ring_find(&self, key: u64) -> Option<u8> {
        let len = self.ring_len as usize;
        let mut i = 0;
        while i < len {
            if self.ring_hash[i] == key {
                return Some(i as u8);
            }
            i += 1;
        }
        None
    }

    /// Bump the answer histogram: increment an existing entry or insert;
    /// when full, evict the minimum-count slot (ties → lowest index —
    /// deterministic, no HashMap iteration anywhere).
    fn bump_hist(&mut self, key: u64) {
        let hist_len = self.hist_len as usize;
        let mut i = 0;
        while i < hist_len {
            if self.hist_hash[i] == key {
                self.hist_count[i] = self.hist_count[i].saturating_add(1);
                return;
            }
            i += 1;
        }
        if hist_len < RING_CAPACITY {
            self.hist_hash[hist_len] = key;
            self.hist_count[hist_len] = 1;
            self.hist_len += 1;
        } else {
            let mut min_i = 0;
            let mut i = 1;
            while i < RING_CAPACITY {
                if self.hist_count[i] < self.hist_count[min_i] {
                    min_i = i;
                }
                i += 1;
            }
            self.hist_hash[min_i] = key;
            self.hist_count[min_i] = 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Vote composition (the third halt-vote family)
// ─────────────────────────────────────────────────────────────────────

/// Compose halt votes from the independent signal families.
///
/// Precedence (documented, deterministic, zero-alloc): ANY `Halt` vote wins
/// over unanimous `Continue`; among halts the EARLIEST step wins; a step tie
/// resolves to the vote that appears FIRST in the slice (family order is the
/// caller's declared precedence). Empty slice → `Continue`.
pub fn compose_votes(votes: &[HaltVote]) -> HaltVote {
    let mut best: Option<HaltVote> = None;
    for &vote in votes {
        // Continue votes never win — only halts propagate.
        let StructuralHaltDecision::Halt { reason, step } = vote else {
            continue;
        };
        match best {
            // Earliest step wins; a step tie keeps the EARLIER slice vote
            // (declared family precedence), so `step >= best_step` is a no-op.
            Some(StructuralHaltDecision::Halt {
                step: best_step, ..
            }) if step >= best_step => {}
            _ => {
                best = Some(StructuralHaltDecision::Halt { reason, step });
            }
        }
    }
    best.unwrap_or(StructuralHaltDecision::Continue)
}

/// Record a numeric-family decision as a structural-family vote.
///
/// The composition point for the PoC: run the numeric arbiter
/// (`GainCostLoopHalter::halt_decision`) and this monitor over the same
/// trace, convert the numeric decision with this function, and merge with
/// [`compose_votes`]. `RefusedFloor` and `Continue` both map to `Continue`
/// (a floor refusal is a continue); any `Halt` maps to
/// [`StructuralHaltReason::NumericArbiter`] — the vote records THAT a
/// foreign family fired and at which step, not the numeric reason detail.
#[cfg(feature = "gain_cost_halt")]
pub fn vote_from_numeric(numeric: crate::gain_cost_halt::HaltDecision, step: usize) -> HaltVote {
    match numeric {
        crate::gain_cost_halt::HaltDecision::Halt { .. } => StructuralHaltDecision::Halt {
            reason: StructuralHaltReason::NumericArbiter,
            step,
        },
        crate::gain_cost_halt::HaltDecision::Continue
        | crate::gain_cost_halt::HaltDecision::RefusedFloor => StructuralHaltDecision::Continue,
    }
}

// ─────────────────────────────────────────────────────────────────────
// T1 mechanics tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    /// Feed a slice of answer strings, returning the decision per step.
    fn run(monitor: &mut StructuralTraceMonitor, answers: &[&str]) -> Vec<StructuralHaltDecision> {
        answers.iter().map(|a| monitor.step(a)).collect()
    }

    // ── normalization ──────────────────────────────────────────────

    #[test]
    fn normalized_hash_is_case_and_whitespace_insensitive() {
        let a = normalized_answer_hash("The Answer  Is 42");
        let b = normalized_answer_hash("  the answer\tis\n42 ");
        let c = normalized_answer_hash("THE ANSWER IS 42");
        assert_eq!(a, b, "case/whitespace/trim must not change the identity");
        assert_eq!(a, c);
        // Internal run collapse: "A  B" == "A B" == "A\tB".
        assert_eq!(
            normalized_answer_hash("A  B"),
            normalized_answer_hash("A B")
        );
        assert_eq!(
            normalized_answer_hash("A  B"),
            normalized_answer_hash("A\tB")
        );
        // The empty answer is a valid identity; whitespace-only answers
        // TRIM TO IT (same normalized stream), while any content differs.
        assert_eq!(normalized_answer_hash("  "), normalized_answer_hash(""));
        assert_ne!(normalized_answer_hash(""), normalized_answer_hash("a"));
    }

    #[test]
    fn normalized_hash_distinguishes_content() {
        let base = normalized_answer_hash("42");
        assert_ne!(base, normalized_answer_hash("43"));
        assert_ne!(base, normalized_answer_hash("4 2"));
        assert_ne!(base, normalized_answer_hash(""));
        // Deterministic across calls.
        assert_eq!(base, normalized_answer_hash("42"));
    }

    // ── transition truth table ─────────────────────────────────────

    #[test]
    fn transition_truth_table() {
        // Row 1: establishment (None → A) is Correct, not a positional change.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        assert_eq!(m.step_key(100), StructuralHaltDecision::Continue);
        assert_eq!(m.last_transition(), Some(StructuralTransition::Correct));
        assert_eq!(
            m.verify_run(),
            0,
            "establishment does not start a verify run"
        );

        // Row 2: same answer → Verify, run increments.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = m.step_key(100);
        let _ = m.step_key(100);
        assert_eq!(m.last_transition(), Some(StructuralTransition::Verify));
        assert_eq!(m.verify_run(), 1);

        // Row 3: fresh answer → Correct (push), run resets.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = m.step_key(100);
        let _ = m.step_key(100);
        let _ = m.step_key(200);
        assert_eq!(m.last_transition(), Some(StructuralTransition::Correct));
        assert_eq!(m.verify_run(), 0);

        // Row 4: revisit of an abandoned ring entry → BacktrackRevisit.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = m.step_key(100);
        let _ = m.step_key(200);
        let _ = m.step_key(100);
        assert_eq!(
            m.last_transition(),
            Some(StructuralTransition::BacktrackRevisit)
        );
        assert_eq!(m.revisit_count(), 1);

        // Row 5 (defensive): re-activation of a never-abandoned entry is
        // Correct. Reachable only via internal state surgery, so exercise
        // the flag directly: push 100 (slot 0, current, not abandoned), then
        // make 200 evict nothing but take over as current without marking
        // 100 abandoned — i.e. drive the observe path with a hand-crafted
        // current so 100 is in-ring, non-current, abandoned == 0.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = m.step_key(100); // slot 0, current
        // Simulate the impossible state: 100 no longer current but flag 0.
        m.current_key = Some(200);
        m.current_slot = 1;
        m.ring_len = 2; // slot 1 "exists" (hash 0 ≠ 100)
        let d = m.step_key(100);
        assert_eq!(d, StructuralHaltDecision::Continue);
        assert_eq!(m.last_transition(), Some(StructuralTransition::Correct));
        assert_eq!(m.revisit_count(), 0, "the defensive arm is not a revisit");
    }

    // ── ring wraparound ────────────────────────────────────────────

    #[test]
    fn ring_wraparound_evicts_oldest() {
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        // 9 distinct answers: ring capacity 8 → answer 1 (the oldest slot)
        // is evicted when answer 9 arrives; answers 2..=9 stay resident.
        for k in 1..=9u64 {
            let _ = m.step_key(k);
        }
        assert_eq!(m.last_transition(), Some(StructuralTransition::Correct));
        // Re-propose answer 1 (evicted) → fresh Correct, NOT a revisit.
        let _ = m.step_key(1);
        assert_eq!(
            m.last_transition(),
            Some(StructuralTransition::Correct),
            "an evicted answer is fresh again (bounded-memory semantics)"
        );
        assert_eq!(m.revisit_count(), 0);
        // Re-propose a STILL-RESIDENT answer → revisit. Key 1's fresh insert
        // above consumed the write slot (evicting key 2), so the probe must
        // sit deeper in the ring: key 5 (slot 4) was left at step 6 and is
        // untouched by any wraparound write. The revisit path never
        // inserts, so the probe cannot evict anything either.
        let _ = m.step_key(5);
        assert_eq!(
            m.last_transition(),
            Some(StructuralTransition::BacktrackRevisit)
        );
    }

    // ── policies ───────────────────────────────────────────────────

    #[test]
    fn self_loop_fires_at_exactly_k() {
        // K=2: [A, A] — one proposal + one verify (run 1) → Continue.
        // [A, A, A] — the K-th verify → Halt at step 3. Never earlier.
        let mut m = StructuralTraceMonitor::new(SelfLoopHalt::paper_default().into());
        let decisions = run(&mut m, &["A", "A", "A", "A"]);
        assert_eq!(decisions[0], StructuralHaltDecision::Continue);
        assert_eq!(
            decisions[1],
            StructuralHaltDecision::Continue,
            "run=1 < K=2"
        );
        assert_eq!(
            decisions[2],
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::SelfLoop,
                step: 3
            },
            "halt lands exactly on the K-th consecutive verify"
        );
        // Frozen: the post-halt step replays the recorded decision.
        assert_eq!(decisions[3], decisions[2]);
        assert!(m.is_halted());
    }

    #[test]
    fn self_loop_k_one_fires_on_first_verify() {
        let mut m = StructuralTraceMonitor::new(SelfLoopHalt::new(1).into());
        let decisions = run(&mut m, &["A", "A"]);
        assert_eq!(decisions[0], StructuralHaltDecision::Continue);
        assert_eq!(
            decisions[1],
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::SelfLoop,
                step: 2
            }
        );
    }

    #[test]
    fn self_loop_does_not_fire_while_answers_change() {
        let mut m = StructuralTraceMonitor::new(SelfLoopHalt::paper_default().into());
        for d in run(&mut m, &["A", "B", "A", "B", "C", "D"]) {
            assert_eq!(d, StructuralHaltDecision::Continue);
        }
    }

    #[test]
    fn backtrack_revisit_fires_on_first_revisit() {
        let mut m = StructuralTraceMonitor::new(BacktrackRevisitHalt.into());
        let decisions = run(&mut m, &["A", "B", "A"]);
        assert_eq!(decisions[0], StructuralHaltDecision::Continue);
        assert_eq!(decisions[1], StructuralHaltDecision::Continue);
        assert_eq!(
            decisions[2],
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::BacktrackRevisit,
                step: 3
            }
        );
    }

    #[test]
    fn revisit_refires_after_re_abandonment() {
        // With Never (observation only), A,B,A,B,A → 3 revisit events: each
        // re-activation of A clears its abandoned flag, so re-abandoning A
        // re-arms the cycle for the next revisit.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = run(&mut m, &["A", "B", "A", "B", "A"]);
        assert_eq!(m.revisit_count(), 3);
    }

    #[test]
    fn never_policy_never_halts() {
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        for k in 0..64u64 {
            let d = m.step_key(k % 4);
            assert_eq!(d, StructuralHaltDecision::Continue);
        }
        assert!(!m.is_halted());
    }

    #[test]
    fn post_halt_freezes_and_reset_clears() {
        let mut m = StructuralTraceMonitor::new(SelfLoopHalt::new(1).into());
        let _ = m.step_key(7);
        let halt = m.step_key(7);
        assert!(halt.is_halt());
        // Frozen: further steps (even changing answers) replay the halt.
        assert_eq!(m.step_key(9), halt);
        assert_eq!(m.step_key(7), halt);
        assert_eq!(m.step_count(), 2, "frozen steps do not consume state");
        // Reset: fresh episode, same policy.
        m.reset();
        assert!(!m.is_halted());
        assert_eq!(m.step_count(), 0);
        assert_eq!(m.step_key(7), StructuralHaltDecision::Continue);
    }

    #[test]
    fn text_and_key_paths_agree() {
        // The text path is exactly the key path over normalized hashes.
        let mut a = StructuralTraceMonitor::new(HaltPolicy::Never);
        let mut b = StructuralTraceMonitor::new(HaltPolicy::Never);
        let text = ["Foo  BAR", "baz", "foo bar"];
        let keys: Vec<u64> = text.iter().map(|s| normalized_answer_hash(s)).collect();
        for (ta, &ka) in text.iter().zip(&keys) {
            assert_eq!(a.step(ta), b.step_key(ka));
        }
        // Normalization made "Foo  BAR" and "foo bar" the same identity.
        assert_eq!(keys[0], keys[2]);
    }

    // ── T3 classifier mapping gate ─────────────────────────────────

    #[test]
    fn classify_empty_trace_defaults_to_late_k2() {
        let m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::LateLanding);
        assert_eq!(c.policy, HaltPolicy::SelfLoop(SELF_LOOP_DEFAULT_K));
    }

    #[test]
    fn classify_single_answer_trace_is_late_landing() {
        // A,A,A: no positional change ever (establishment ≠ change) → the
        // whole trace is a verify tail; purity 1.0 ≥ 0.75 → K=2.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = run(&mut m, &["A", "A", "A"]);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::LateLanding);
        assert_eq!(c.policy, HaltPolicy::SelfLoop(2));
    }

    #[test]
    fn classify_two_answer_oscillation_is_explorer_despite_purity_half() {
        // A,B,A,B: histogram {A:2, B:2} → purity exactly 0.5, yet the tail
        // is zero (a change landed on the last step). Positional mass must
        // outrank purity for the PATTERN selection.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = run(&mut m, &["A", "B", "A", "B"]);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::Explorer);
        assert_eq!(c.policy, HaltPolicy::BacktrackRevisit);
    }

    #[test]
    fn classify_late_landing_k3_when_landing_contested() {
        // A,B,C,X,X,X,X,X (8 steps): tail = 4/8 ≥ 1/2 → Late. Histogram
        // {X:5, A:1, B:1, C:1} → purity = (25+3)/64 = 0.4375 < 0.75 → K=3.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = run(&mut m, &["A", "B", "C", "X", "X", "X", "X", "X"]);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::LateLanding);
        assert_eq!(c.policy, HaltPolicy::SelfLoop(LATE_LANDING_DEFAULT_K));
    }

    #[test]
    fn classify_late_landing_k2_when_landing_dominant() {
        // A then X ×9 (11 steps): purity = (100+1)/121 ≈ 0.835 ≥ 0.75 → K=2.
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let mut trace = vec!["A"];
        trace.extend(std::iter::repeat_n("X", 9));
        let _ = run(&mut m, &trace);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::LateLanding);
        assert_eq!(c.policy, HaltPolicy::SelfLoop(SELF_LOOP_DEFAULT_K));
    }

    #[test]
    fn classify_all_distinct_is_explorer() {
        let mut m = StructuralTraceMonitor::new(HaltPolicy::Never);
        let _ = run(&mut m, &["A", "B", "C", "D", "E"]);
        let c = m.classify_prefix();
        assert_eq!(c.pattern, Pattern::Explorer);
        assert_eq!(c.policy, HaltPolicy::BacktrackRevisit);
    }

    // ── Auto fusion end-to-end ─────────────────────────────────────

    #[test]
    fn auto_fusion_halts_explorer_on_revisit() {
        let mut m = StructuralTraceMonitor::auto();
        let decisions = run(&mut m, &["A", "B", "A"]);
        assert_eq!(decisions[0], StructuralHaltDecision::Continue);
        assert_eq!(decisions[1], StructuralHaltDecision::Continue);
        assert_eq!(
            decisions[2],
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::BacktrackRevisit,
                step: 3
            },
            "Auto resolves Explorer → backtrack policy → the cycle halts it"
        );
    }

    #[test]
    fn auto_fusion_stays_quiet_while_the_tail_is_under_half() {
        // Late landing: A,B,C then a verify tail. Auto stays Explorer while
        // the tail fraction is below 1/2 — all 7 steps Continue even though
        // the verify run already reached 3 at step 7 (the tail there is
        // 3/7 < 1/2, so the pattern has not flipped yet).
        let mut m = StructuralTraceMonitor::auto();
        let decisions = run(&mut m, &["A", "B", "C", "X", "X", "X", "X"]);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(*d, StructuralHaltDecision::Continue, "step {}", i + 1);
        }
    }

    #[test]
    fn auto_fusion_late_landing_halt_step_is_pinned() {
        // The same trace + one more verify: the halt landed ON step 8 and
        // the frozen episode replays it for every later step.
        let mut m = StructuralTraceMonitor::auto();
        let decisions = run(&mut m, &["A", "B", "C", "X", "X", "X", "X", "X", "X"]);
        assert_eq!(
            decisions[7],
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::SelfLoop,
                step: 8
            },
            "the tail fraction crosses 1/2 at step 8 and K=3 is already covered"
        );
        assert_eq!(decisions[8], decisions[7], "frozen replay");
    }

    // ── vote composition ───────────────────────────────────────────

    #[test]
    fn compose_votes_precedence() {
        use StructuralHaltReason::*;
        // Empty → Continue.
        assert_eq!(compose_votes(&[]), StructuralHaltDecision::Continue);
        // Unanimous continue.
        assert_eq!(
            compose_votes(&[
                StructuralHaltDecision::Continue,
                StructuralHaltDecision::Continue
            ]),
            StructuralHaltDecision::Continue
        );
        // Any halt wins.
        assert_eq!(
            compose_votes(&[
                StructuralHaltDecision::Continue,
                StructuralHaltDecision::Halt {
                    reason: SelfLoop,
                    step: 5
                },
            ]),
            StructuralHaltDecision::Halt {
                reason: SelfLoop,
                step: 5
            }
        );
        // Earliest step wins regardless of family or slice position.
        assert_eq!(
            compose_votes(&[
                StructuralHaltDecision::Halt {
                    reason: NumericArbiter,
                    step: 4
                },
                StructuralHaltDecision::Halt {
                    reason: BacktrackRevisit,
                    step: 7
                },
            ]),
            StructuralHaltDecision::Halt {
                reason: NumericArbiter,
                step: 4
            }
        );
        // Step tie → first vote in the slice (declared family precedence).
        assert_eq!(
            compose_votes(&[
                StructuralHaltDecision::Halt {
                    reason: BacktrackRevisit,
                    step: 5
                },
                StructuralHaltDecision::Halt {
                    reason: NumericArbiter,
                    step: 5
                },
            ]),
            StructuralHaltDecision::Halt {
                reason: BacktrackRevisit,
                step: 5
            }
        );
    }

    #[cfg(feature = "gain_cost_halt")]
    #[test]
    fn vote_from_numeric_bridge() {
        use crate::gain_cost_halt::HaltDecision;
        assert_eq!(
            vote_from_numeric(HaltDecision::Continue, 3),
            StructuralHaltDecision::Continue
        );
        assert_eq!(
            vote_from_numeric(HaltDecision::RefusedFloor, 3),
            StructuralHaltDecision::Continue,
            "a floor refusal is a continue"
        );
        assert_eq!(
            vote_from_numeric(
                HaltDecision::Halt {
                    reason: crate::gain_cost_halt::HaltReason::Oscillation
                },
                9
            ),
            StructuralHaltDecision::Halt {
                reason: StructuralHaltReason::NumericArbiter,
                step: 9
            }
        );
    }

    // ── layout guards ──────────────────────────────────────────────

    #[test]
    fn size_bounds_stay_cache_friendly() {
        // The decision carries a usize payload (step) — 16 B is the floor
        // for (tag, u8, usize) niche packing.
        assert!(
            size_of::<StructuralHaltDecision>() <= 16,
            "StructuralHaltDecision grew to {} B",
            size_of::<StructuralHaltDecision>()
        );
        assert_eq!(size_of::<StructuralHaltReason>(), 1);
        assert_eq!(size_of::<StructuralTransition>(), 1);
        // The monitor is fixed-array state only — no heap, no Vec. Guard
        // against accidental growth (it is constructed per trace/episode).
        assert!(
            size_of::<StructuralTraceMonitor>() <= 512,
            "StructuralTraceMonitor grew to {} B",
            size_of::<StructuralTraceMonitor>()
        );
    }

    #[test]
    fn ring_capacity_is_eight() {
        assert_eq!(RING_CAPACITY, 8);
        assert_eq!(SelfLoopHalt::paper_default().k, 2);
        assert_eq!(SelfLoopHalt::default(), SelfLoopHalt { k: 2 });
        // Clamped constructor.
        assert_eq!(SelfLoopHalt::new(0).k, 1);
        assert_eq!(SelfLoopHalt::new(5).k, 5);
    }
}
