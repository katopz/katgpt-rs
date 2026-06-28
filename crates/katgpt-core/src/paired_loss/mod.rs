//! `paired_loss` — Paired Token-Level Loss Gap Diagnostic (Plan 335).
//!
//! Generic, modelless, zero-alloc A/B measurement primitive distilled from
//! Li & Merrill 2026 (arXiv:2606.20936, AI2) — *Comparing Transformers and
//! Hybrid Models at the Token Level*. See Research 319 for the full
//! distillation.
//!
//! # The core idea (one paragraph)
//!
//! Given two log-probability traces over the SAME prefixes (e.g., two
//! adapters, HLA-on vs HLA-off, two router configs), compute the per-token
//! gap `Δ_i = ℓ_A[i] − ℓ_B[i]`, stratify by token class (Content/Function/
//! Other/BracketOpen/BracketClose/CopyN), and report filtered aggregates
//! (`ALL_TOKENS` / `TOP-K∩NO-COPY` / `COPY-N-ONLY`) that amplify small
//! architecture gaps aggregate loss hides. The paper shows `TOP-K∩NO-COPY`
//! roughly doubles the Transformer–Hybrid separation vs `ALL_TOKENS` on 1B
//! pretraining runs (Figure 7).
//!
//! Companion theoretical tool: **Proposition 1** (`DKL(p⋆_τ ‖ p_ϕ,τ) ≤
//! log|V_τ|`) — exposed as [`ClassSizeBound`], it annotates *which* token
//! classes have room for a richer feature map to help (large `log|V_τ|`) vs
//! which are structurally bounded (small `log|V_τ|`). This is the
//! information-theoretic justification of our raw-vs-latent sync boundary
//! (physical domain: small `V_τ` → raw sufficient; semantic domain: large
//! `V_τ` → latent earns its keep). See Research 319 §2.2.
//!
//! # Modelless (katgpt-rs mandate)
//!
//! Pure forward-pass analysis — no training, no backprop, no gradient
//! descent. The diagnostic operates on log-probability *traces* (outputs of
//! forward passes), not on weights or gradients. It's a post-hoc measurement
//! tool, not an inference mechanism (Research 319 §3: NOT Super-GOAT).
//!
//! # Sigmoid not softmax (AGENTS.md)
//!
//! N/A — this primitive has no gates. It's pure subtract + sum + log. There
//! is no `sigmoid` or `softmax` symbol anywhere in this module. The
//! diagnostic is a measurement tool; the consumer's gates live elsewhere.
//!
//! # Latent vs Raw (AGENTS.md)
//!
//! - [`PairedLossGap::deltas`] → raw (output of forward passes; the consumer
//!   owns the raw-vs-latent decision upstream).
//! - [`ClassSizeBound::log_v_tau`] → raw (closed-form `log(V_τ)`; a constant
//!   annotation, not synced state).
//! - [`TokenClass`] → raw (a tag label; consumer-side metadata).
//!
//! No data crosses a sync boundary — this is a local measurement tool.
//!
//! # Zero-allocation discipline
//!
//! - [`PairedLossGap::from_log_probs`] allocates the delta vec ONCE
//!   (`Vec::with_capacity(L)`) — this IS the output, not a hot-path alloc.
//! - All query methods (`mean_gap`, `mean_gap_for_class`, `filtered_mean`)
//!   are `&self` and use iterator folds — **zero heap allocation** on the
//!   query hot path. No mask `Vec`, no intermediate collections.
//! - G2 gate (Phase 2 T2.1): `< 1µs` for `from_log_probs + filtered_mean` at
//!   L=8192 (one subtract + one SIMD sum + one masked fold).
//!
//! # What this primitive is NOT
//!
//! - NOT an inference mechanism — it doesn't generate tokens, route compute,
//!   or mutate weights. It measures the gap between two existing forward
//!   passes. (Research 319 §3: NOT Super-GOAT.)
//! - NOT a regression tool — the paper ships a full OLS regression with
//!   controls (difficulty, frequency, position, subword status, local reuse,
//!   previous-token distance, token frequency). That's a research-grade
//!   statistical tool for the paper's claims; this primitive ships the raw
//!   tag-stratified means + filtered aggregates (the high-signal subset).
//!   See Plan 335 Design Notes + Research 319 §5 R1.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/335_paired_loss_gap_diagnostic_primitive.md`]
//! - Research: [`katgpt-rs/.research/319_Paired_Token_Loss_Gap_Discourse_State_Diagnostic.md`]
//! - Source paper: [arXiv:2606.20936](https://arxiv.org/abs/2606.20936) —
//!   Li & Merrill, AI2, Jun 2026.
//! - Theoretical predecessor: Research 242 (Mozer et al. 2026,
//!   arXiv:2604.17121 — topological state-tracking diagnosis).

mod gap;
mod tagger;
#[cfg(test)]
mod tests;
mod types;

pub use tagger::{CopyNGramTagger, TokenTagger};
pub use types::{
    ClassGapReport, ClassGapRow, ClassSizeBound, FilterKind, FilterScratch, PairedLossGap,
    TokenClass,
};
