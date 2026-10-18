//! AC-GPT Arbitrary-Conditional Prefix — modelless mask builder + sequence
//! augmenter that turns any causal Transformer forward pass into a single-pass
//! arbitrary-conditional forward pass `p(xe | xc)`, including conditioning on
//! **future** tokens.
//!
//! Distilled from Lu, Elmoznino, Gagnon, Mittal, Kasetty, Lajoie. *Simplifying
//! the Modeling of Arbitrary Conditionals in Natural Language* (AC-GPT).
//! [arXiv:2606.14943](https://arxiv.org/abs/2606.14943). Mila / McGill /
//! Université de Montréal. 12 Jun 2026.
//!
//! - **Plan:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md`
//! - **Research:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md`
//!
//! # Mechanism (the three-region attention rule)
//!
//! Given a base token sequence `x = xc ∪ xe` (in original order) and a sorted
//! set of conditioning positions `xc ⊂ {0..|x|}`, the augmented sequence is
//!
//! ```text
//! ┌────────────────────────┬─────────────────────────────────────┐
//! │  xc copies (front)     │  full sequence x = xc ∪ xe          │
//! │  region r0             │  region r1                          │
//! │  bidirectional self-   │  causal attention everywhere        │
//! │  attention among copies│  loss only on xe                    │
//! └────────────────────────┴─────────────────────────────────────┘
//! ```
//!
//! The attention mask over the augmented sequence obeys the **three-region
//! rule** — the load-bearing leakage-prevention discipline from the paper:
//!
//! - `(i ∈ r0, j ∈ r0) → true`  — copies attend to each other bidirectionally
//!   (this is what lets copies sync across layers without leaking through the
//!   in-place originals).
//! - `(i ∈ r1, j ∈ r0) → true`  — eval positions attend to all copies.
//! - `(i ∈ r0, j ∈ r1) → false` — copies do NOT attend back to the original
//!   sequence; they ARE the original tokens, and attending back would create a
//!   duplicate-key information path.
//! - `(i ∈ r1, j ∈ r1) → original_pos(i) >= original_pos(j)` — standard causal
//!   in the eval region.
//!
//! **Why the copy is necessary (not just letting later eval tokens attend to
//! the in-place conditioning tokens):** the paper's worked example is that
//! without the copy, `x2 → x3 → x1` over two layers leaks future information
//! from `x2` to `x1` *through* the conditioning token `x3`. The copy at the
//! front with bidirectional self-attention among copies (and no attention back
//! to the originals) is what prevents the leakage.
//!
//! # Modelless discipline
//!
//! This primitive operates on whatever causal Transformer already ships (GPT-2
//! small, micro_dllm, game configs). No new weights, no training, no backprop.
//! The "fine-tune Qwen3 / LLaMA with LoRA" recipe from the paper redirects to
//! riir-train; what stays here is the runtime construction.
//!
//! # Performance contract
//!
//! - [`AcPrefix::attends`] is branch-free (boolean `&` / `|` composition, no
//!   short-circuit) and zero-allocation. Suitable for tight inner loops.
//! - [`AcPrefix::original_positions_into`] and [`AcPrefix::loss_mask_into`]
//!   write into caller-provided buffers (zero allocation inside).
//! - [`AcPrefixMask::materialize_from`] is the only allocating call — it
//!   bit-packs the `attends` rule into a `Box<[u64]>` once per augmented
//!   sequence for batched attention kernels that want a materialized mask.
//!
//! # Status
//!
//! Phase 1 (this module): types + bit math, no attention kernel dep.
//! DEFAULT-ON (Plan 313 Phase 4 + Issue 003 Phase 0 RESOLVED, 2026-06-24):
//! G1 0.0 diff vs iterative-MLM (modelless `attends_dedup` unblock eliminates
//! the doubled-signal bias), G2 27.258x speedup, G3 0 mismatches, G4 0 allocs.
//! Multi-layer equivalence remains a non-blocking riir-train follow-up.
//!
//! AC-GPT Arbitrary-Conditional Prefix — modelless mask builder + sequence
//! augmenter that turns any causal Transformer forward into a single-pass
//! arbitrary-conditional forward p(xe | xc) via position-aware copies of xc at
//! the front and a [xc-bidirectional | causal-everywhere-else] attention mask
//! (Lu et al., Mila, arXiv:2606.14943, Plan 313, Research 295). Phase 1 ships
//! types + bit math only — no attention kernel dep, no SVD. DEFAULT-ON (Plan 313, 2026-06-24): G1–G4 PASS via modelless Path 2 (`attends_dedup`) — see .benchmarks/313_ac_prefix_modelless.md. Multi-layer equivalence remains a non-blocking riir-train follow-up.
//!
//! The keyed-Gumbel verify-sampling pair (position-keyed noise — the
//! order-invariant shared stream drafter proposals and target verify
//! samples consume; Plan-614 substrate row) plus the truncated-sampling
//! triple the Phase-2 target side composes (mask → scale → keyed argmax)
//! and the posterior inverse sampler (Issue 918).

mod forward;
mod types;

pub use forward::ForwardForAcPrefix;
pub use types::{AcPrefix, AcPrefixMask};
// The keyed-Gumbel verify-sampling pair (position-keyed noise — the
// order-invariant shared stream drafter proposals and target verify
// samples consume; Plan-614 substrate row) + the posterior inverse
// sampler (Issue 918: reconstruct consistent noise FROM a realized pick,
// max-first truncated construction).
pub use types::{
    keyed_gumbel_max_sample, keyed_gumbel_max_sample_masked, keyed_gumbel_max_sample_truncated,
    keyed_gumbel_noise, keyed_posterior_gumbel_noise, truncation_keep_mask,
};

pub(crate) use types::gumbel_max_sample;
