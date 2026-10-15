# Issue 941: Lift the Token/Latent Position-Provenance Convention Into the SwiR Host Contract

**Status:** LANDED 2026-10-16 — contract lifted (PositionProvenance + ProvenanceLedger + DecodeBackend provenance doc + 6 contract tests, all green); the riir-ai gemma2 backend stays the reference implementation
**Claimed:** shikuwa-4090-agent, 1792100393, expires 2026-10-17T22:00:00Z
**Date:** 2026-10-15
**Source:** katgpt-rs/.research/622 (HybridAgent distill, arXiv:2610.18861 §"continuous working memory"); corrected by verdict review round 1 — the original "not shipped / P1 bug" framing was a substrate-first miss

## What already ships

`riir-ai/crates/riir-engine/src/swir_validation/gemma2_backend.rs` (the Plan 313 real-model host) already implements the core discipline:

```rust
// Track the token being fed (if Explicit mode) for future penalty.
// For Latent mode (soft embedding), we don't track a specific token.
if let Some(id) = token_id {
    self.generated_tokens.push(id);
}
```

Soft-embedding positions (`token_id = None`) are excluded from the penalty history — HybridAgent's `kv_ledger` semantics, under a different name. What's missing is everything around it:

## Deliverables

- [x] Name it and contract-ize it katgpt-rs-side: `PositionProvenance::Tok(id) | Lat { span, step }` as part of the swir host contract (StepContext / DecodeBackend documentation + a type the host can carry), with the materialized view `M(L)` (Tok ids only) as the single history surface samplers read.
- [x] Pin the riir-engine Gemma2 backend as the REFERENCE implementation with a cross-repo citation in the contract docs (consume, never fork).
- [x] Contract tests in katgpt-rs: with no latent spans, provenance-on behavior is bit-identical to provenance-off; a latent span's argmaxes never enter the penalty set; a real repeated token always does.
- [x] Purity invariant stated in the contract: the ledger is metadata, never an attention mask (latent positions attend and are attended — HybridAgent's own design).
- [x] Cross-ref: the two LIVE defects found in the same subsystem during verdict review are filed riir-ai-side (riir-ai Issue 1075: penalty compounding over duplicate history entries + latent pseudo-argmaxes leaking into the bench scoring path) — this issue owns only the katgpt-rs contract lift.

## Game note

The ledger is the two-brain law at cache level: `Tok` positions persist (info brain, syncable); `Lat` positions are think-brain scratch that never serializes — the runtime expression of the latent/raw sync-boundary rule.

## Result (2026-10-16)

Landed via the isolated worktree (the main worktree is wedged by a sibling's
dirty Cargo.toml - no rebase possible; worktree at origin/develop):

- `crates/katgpt-transformer/src/swir/position_provenance.rs` -
  `PositionProvenance::{Tok(u32), Lat{span, step}}` + `ProvenanceLedger`
  (`record` / `materialized()` = M(L) / `position_count` / `latent_positions` /
  `latent_spans` / `clear`). Lat content is structurally unretainable (counts
  only - "never serializes" is enforced by the ABSENCE of any Lat-content
  field, not by convention).
- `swir/mod.rs` - module registered + re-exported + the module-doc surface note.
- `swir/bench.rs` - `DecodeBackend::decode_step` doc gained the
  "# Position-provenance contract (Issue 941)" section: `Some(id)` MUST enter
  M(L); `None` (soft-embedding) MUST NOT; reference implementation = riir-ai
  `swir_validation/gemma2_backend.rs` (Plan 313 host); ledger is metadata,
  NEVER an attention mask.
- Purity invariant stated in the position_provenance module docs.
- 6 contract tests green at `--features swir_switch_thinking` (51/51 swir
  module tests): no-latent bit-identity vs plain push history; latent argmaxes
  never enter the penalty set (M(L) compresses them out); real repeated tokens
  always penalized; span-change counting; latent interleaving leaves M(L)
  bit-identical; clear() full reset. Clippy clean at the posture.
- No new feature flag - rides the existing opt-in `swir_switch_thinking`.
- riir-ai Issue 1075 cross-ref carried in the module docs.
