//! # Head Substitution Gate — IoU + FaithfulnessProbe wrapper for FuncAttn.
//!
//! **Plan 353** (Gain-tier, opt-in). Source paper:
//! [arXiv:2606.19317](https://arxiv.org/abs/2606.19317) — Hayes, Li, Andreas,
//! *Explaining Attention with Program Synthesis*, MIT CSAIL / NJIT, 30 Jun 2026.
//!
//! ## What this module ships (and what it deliberately does NOT)
//!
//! This module ships **one** small piece: [`HeadSubstitutionGate`] — a decision
//! struct that decides when a [`FuncAttn`](crate::funcattn)-style surrogate
//! should substitute for a real attention head on the current forward pass.
//!
//! It does **not** ship a new primitive. The original Plan 353 draft proposed a
//! `ProgramSynthesizedHead` primitive + `Box<dyn SynthesizedAttentionFn>` trait.
//! A user-prompted re-review identified that:
//!
//! 1. [`FuncAttn`](crate::funcattn) (Plan 286, Research 257) already ships the
//!    `tokens → attention via external operator` primitive shape. The proposed
//!    `SynthesizedAttentionFn` trait is structurally `dyn FuncAttnKernel` —
//!    redundant.
//! 2. `katgpt-percepta` (Plan 064) already ships the programs-as-attention
//!    paradigm.
//!
//! So the redundant primitive was dropped. What remains is the **control loop**
//! that gates when the existing primitive fires, using the paper's empirical
//! finding that IoU `r > 0.9` correlates with substitution cost (paper §3
//! Fig 5b). That control loop is small enough that the plan's own Risk note
//! flags it: *the gate may be too thin to justify a feature flag*. The honest
//! answer is "yes, barely" — see the Risk note in [`gate::HeadSubstitutionGate`].
//!
//! ## The cadence pattern (Plan 287 SinkAware)
//!
//! Substitution decisions use a two-stage cadence:
//!
//! 1. **Cheap proxy (per-call)**: [`iou()`] between the real head's attention
//!    row and the surrogate. Below `tau_iou`, reject immediately. This is the
//!    paper's headline empirical finding — IoU is a fast, well-correlated
//!    proxy for the expensive causal substitution cost.
//! 2. **Expensive veto (cached, audit cadence)**: [`FaithfulnessProfile`]
//!    re-measured every N ticks (NOT per-token) and cached. The cached
//!    [`gate::worst_case_behavior_delta`] acts as the veto: if the head is
//!    causally load-bearing under any intervention variant, substitution is
//!    rejected.
//!
//! This split is the "Plan 287 SinkAware" pattern applied to head substitution:
//! expensive diagnostics run on a slow cadence and feed a cheap hot-path
//! decision. The hot path
//! ([`HeadSubstitutionGate::should_substitute`]) is alloc-free and branch-light.
//!
//! ## GOAT verdict (Gain-tier)
//!
//! - **G1 (correctness)**: PASS — [`iou()`] hand-computed cases (identity,
//!   disjoint, partial-overlap, all-zero) + gate decision cases (identity
//!   accept, disjoint reject, partial-overlap boundary, faithfulness veto).
//! - **G2 (IoU→delta correlation, synthetic)**: PASS — Spearman ρ ≤ −0.9
//!   reproduced on the synthetic noise-blend harness (paper's `r > 0.9`).
//!   **Real-head G2 is DEFERRED to riir-ai** — it requires a real transformer
//!   forward pass, which lives outside this crate.
//! - **G3 (hot-path latency)**: PASS — `should_substitute` overhead is
//!   negligible vs the always-false baseline (a single comparison + a cached
//!   slice index).
//! - **G4 (zero-alloc)**: PASS — `#[inline]` on `should_substitute`; no Vec
//!   growth, no Box on the hot path. Allocation happens only on the
//!   audit-cadence refresh path ([`HeadSubstitutionGate::refresh_cache`] /
//!   [`HeadSubstitutionGate::update_head`]).
//!
//! **Stays opt-in.** Gain-tier primitives are not promoted to default-on
//! unless a fusion upgrades them.
//!
//! ## Adaptation note — `FaithfulnessProfile<D>` surface
//!
//! The plan's pseudocode referenced a non-existent field
//! `behavior_delta_when_replaced`. The real [`FaithfulnessProfile<D>`]
//! (in [`crate::faithfulness::types`]) exposes four deltas:
//! `empty_delta`, `shuffle_or_corrupt_delta`, `irrelevant_delta`,
//! `filler_delta`. There is no single "behavior delta when replaced" field.
//!
//! The gate uses [`gate::worst_case_behavior_delta`] — the max of the three
//! *disruptive* intervention deltas (`shuffle_or_corrupt`, `irrelevant`,
//! `filler`), excluding `empty_delta` (the graceful-absence baseline, which is
//! small for a faithful consumer and not a substitution-cost signal). This is
//! the conservative reading: if **any** disruption produces a large behavioral
//! delta, the head is causally load-bearing and substitution is vetoed.
//!
//! [`FaithfulnessProfile<D>`]: crate::faithfulness::types::FaithfulnessProfile
//! [`FaithfulnessProfile`]: crate::faithfulness::types::FaithfulnessProfile
//!
//! Plan 353 — Head Substitution Gate (Gain-tier, opt-in). Small decision
//! struct that decides when a FuncAttn-style surrogate should substitute for
//! a real attention head, using the paper's IoU cheap-proxy (§3 Fig 5b r>0.9)
//! + cached FaithfulnessProfile veto (Plan 287 SinkAware cadence). NOT a new
//! primitive — the original draft proposed a redundant ProgramSynthesizedHead
//! primitive that was dropped after re-review identified FuncAttn (above) as
//! the existing primitive surface. Stays opt-in: Gain-tier, and the plan's
//! own Risk note flags it as borderline-thin for a feature flag.

pub mod gate;
pub mod iou;

// Support-instability regime detection (Issue 693 / Research 513 /
// arXiv:2608.22764 LpWM) — the consecutive-tick TEMPORAL consumer of `iou`,
// + mode-factored state accessors (SupportMask / magnitudes). Opt-in behind
// `support_regime` (which implies this crate's own feature gate — the
// module lives beside `iou`).
#[cfg(feature = "support_regime")]
pub mod support_instability;
#[cfg(feature = "support_regime")]
pub use support_instability::{
    SupportInstabilityDetector, SupportMask, magnitudes, support_instability,
};

pub use gate::{HeadSubstitutionGate, worst_case_behavior_delta};
pub use iou::iou;
