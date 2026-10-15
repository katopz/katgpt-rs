# Issue 941: Lift the Token/Latent Position-Provenance Convention Into the SwiR Host Contract

**Status:** Open — P2 substrate-first lift (the behavior SHIPS host-side; the contract and its katgpt-rs-side tests do not)
**Claimed:** shikuwa-4090-agent, epoch 2026-10-15T21:39:53Z, expires 2026-10-17T22:00:00Z
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

- [ ] Name it and contract-ize it katgpt-rs-side: `PositionProvenance::Tok(id) | Lat { span, step }` as part of the swir host contract (StepContext / DecodeBackend documentation + a type the host can carry), with the materialized view `M(L)` (Tok ids only) as the single history surface samplers read.
- [ ] Pin the riir-engine Gemma2 backend as the REFERENCE implementation with a cross-repo citation in the contract docs (consume, never fork).
- [ ] Contract tests in katgpt-rs: with no latent spans, provenance-on behavior is bit-identical to provenance-off; a latent span's argmaxes never enter the penalty set; a real repeated token always does.
- [ ] Purity invariant stated in the contract: the ledger is metadata, never an attention mask (latent positions attend and are attended — HybridAgent's own design).
- [ ] Cross-ref: the two LIVE defects found in the same subsystem during verdict review are filed riir-ai-side (riir-ai Issue 1075: penalty compounding over duplicate history entries + latent pseudo-argmaxes leaking into the bench scoring path) — this issue owns only the katgpt-rs contract lift.

## Game note

The ledger is the two-brain law at cache level: `Tok` positions persist (info brain, syncable); `Lat` positions are think-brain scratch that never serializes — the runtime expression of the latent/raw sync-boundary rule.
