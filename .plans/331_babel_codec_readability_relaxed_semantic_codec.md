# Plan 331: BabelCodec — Readability-Relaxed Semantic Codec (Open Primitive)

**Date:** 2026-06-26
**Research:** [katgpt-rs/.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md)
**Source paper:** [arxiv 2606.19857](https://arxiv.org/abs/2606.19857) — BabelTele (Zhu et al., SJTU, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/babel_codec/` (new module) + Cargo feature `babel_codec`
**Status:** Active — Phase 1 unblocking skeleton

---

## Goal

Ship a generic `BabelCodec` trait + two deterministic implementations:

1. **`FixedRuleTextCodec`** — BT-P8 / BT-P13 fixed symbolic mapping rules. Deterministic, BLAKE3-commitable text compression for KG-triple / entity-attribute / config / quest-grammar surfaces. Target: 2–3× compression on Seal corpus (NOT the paper's prompt-elicited 3.6× — we ship the modelless subset).
2. **`SigmoidLatentCodec<D>`** — deterministic dot-product projection + sigmoid gate on `&[f32; D]`. Generic-trait facade over what `DensityBudget` + `extract_hla_slice` (Plan 311) already do for HLA slices — unifies text and latent under one API.

**GOAT gate (must pass before promotion to default):**
- G1 (fidelity): round-trip `decompress(compress(x)) ≡ x` on the deterministic subset (KG triples, entity-attribute pairs, config strings). 100% bit-identical for the fixed-rule inverse.
- G2 (compression): ≥ 2× byte reduction on the real Seal 17k corpus (the one that killed CompressionDrafter, Plan 285/287). Honest target: 2–3×.
- G3 (latency): compress + decompress < 200 ns per message on D=8 latent / < 2 µs per 256-byte text chunk. Plasma-tier budget.
- G4 (no regression): `cargo test -p katgpt-core --all-features` clean.
- G5 (determinism): same input → bit-identical output across ARM64/x86_64/wasm32 (BLAKE3 checks). Required for any future LatCal-commitment path (issue #002).

**Why opt-in until G2 passes:** CompressionDrafter failed G2 twice on Seal (Plan 285/287). BabelCodec must beat that bar on the same corpus before promotion. If G2 fails again, the open primitive stays opt-in as a documented negative result (matching the CompressionDrafter precedent).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/babel_codec/mod.rs` with the `BabelCodec` trait, `BabelPair` struct, and module docs citing Research 312.
- [ ] **T1.2** Add `babel_codec` feature to `crates/katgpt-core/Cargo.toml` (opt-in, empty dep list).
- [ ] **T1.3** Add `babel_codec` to the root `lib.rs` feature-gated module list.

**Exit:** `cargo check -p katgpt-core --features babel_codec` compiles an empty module.

---

## Phase 2 — FixedRuleTextCodec (the modelless text-level primitive)

### Tasks

- [ ] **T2.1** `fixed_rule.rs`: implement BT-P8 schema parser + emitter. Schema (from paper Appendix C.2.8):
  - `S[topic/abbrev]` — section anchor
  - `@entity(K=V)` — entity attribute binding
  - `Config[target]:K=V(unit)` — exact-value config
  - `A>B>C` — pipeline / containment
  - `?[cond]=>[act]` — conditional branch
  - `!obj:detail` — exception
  - `A<>B:conclusion` — comparison
  - Preserve original placeholders (`BIBREF`, `TABREF`) verbatim
  - `NULL` / `?` for missing data
- [ ] **T2.2** Implement `compress(&str) -> Vec<u8>`: tokenize input as `(subject, predicate, object)` triples + entity-attribute pairs + config lines, emit BT-P8 form. Returns compressed bytes (UTF-8 of the BT-P8 string).
- [ ] **T2.3** Implement `decompress(&[u8]) -> String`: parse BT-P8 form back to verbose natural-language-ish triple form. **Deterministic inverse** — round-trip must be bit-identical for the schema-covered subset.
- [ ] **T2.4** Implement `last_ratio()` — compression ratio of the most recent call.
- [ ] **T2.5** Unit tests (≥ 12): round-trip on KG triples, entity-attribute pairs, config strings, conditional branches, comparison matrices, placeholder preservation, NULL handling, nested structures, empty input, max-length input, mixed schema, unicode entity names.
- [ ] **T2.6** Doc example showing before/after on a sample quest dialog.

**Exit:** `cargo test -p katgpt-core --features babel_codec babel_codec::fixed_rule` green.

---

## Phase 3 — SigmoidLatentCodec (the generic latent facade)

### Tasks

- [ ] **T3.1** `sigmoid_latent.rs`: implement `SigmoidLatentCodec<D>` generic over latent dimension. Fields: `directions: [[f32; D]; K]`, `bias: [f32; K]`, `tau: f32`.
- [ ] **T3.2** `compress(&[f32; D]) -> CompressedLatent<K>`: project onto K direction vectors, apply `sigmoid(dot + bias)`, return top-k by magnitude (k ≤ K). **Zero-allocation** — write into a pre-sized `CompressedLatent<K>` scratch buffer.
- [ ] **T3.3** `decompress(reader: &Reader, c: &CompressedLatent<K>) -> [f32; D]`: deterministic pseudo-inverse via the reader's projection matrix. Document that this is a lossy inverse (latent projection is not bijective) — the contract is "recover the top-k subspace", not bit-identical recovery.
- [ ] **T3.4** Cross-reference: doc comment explicitly states this is structurally identical to `DensityBudget` + `extract_hla_slice` (Plan 311) — the value is API uniformity, not new capability.
- [ ] **T3.5** Unit tests (≥ 8): round-trip preserves top-k subspace, zero vector, max-magnitude vector, deterministic across calls, direction orthogonality, bias shift, tau sharpness, K < D case.

**Exit:** `cargo test -p katgpt-core --features babel_codec babel_codec::sigmoid_latent` green.

---

## Phase 4 — BLAKE3 Commitment (for future LatCal bridge)

### Tasks

- [ ] **T4.1** `commitment.rs`: `BabelCommitment` newtype wrapping `[u8; 32]` BLAKE3 digest of the compressed bytes.
- [ ] **T4.2** `commit(&self) -> BabelCommitment` on both codec impls.
- [ ] **T4.3** `verify(&self, commitment: &BabelCommitment) -> bool` — recompute and compare.
- [ ] **T4.4** Unit tests: deterministic across architectures (test on host arch; document that cross-arch is validated in G5 bench, not here), tamper detection, empty input.

**Exit:** `cargo test -p katgpt-core --features babel_codec babel_codec::commitment` green. This unblocks issue #002 (deterministic → LatCal chain commitment).

---

## Phase 5 — GOAT Gate (the gate that killed CompressionDrafter)

### Tasks

- [ ] **T5.1** `tests/bench_331_babel_codec_goat.rs`: G1 round-trip fidelity on 1000 synthetic KG triples + entity-attribute pairs.
- [ ] **T5.2** G2 compression on the **real Seal 17k corpus** (same corpus as Plan 285/287). Measure byte reduction. **Target: ≥ 2×.** Honest expectation: 2–3×.
- [ ] **T5.3** G3 latency: `std::time::Instant` batched median (matching crate convention). Target: < 200 ns / latent msg, < 2 µs / 256-byte text chunk.
- [ ] **T5.4** G4 no-regression: `cargo test -p katgpt-core --all-features` clean.
- [ ] **T5.5** G5 cross-arch determinism: run G1 on ARM64 + x86_64 (wasm32 if feasible), assert bit-identical BLAKE3 commitments.
- [ ] **T5.6** Document results in `katgpt-rs/.benchmarks/331_babel_codec_goat.md`. **Honest negative result if G2 fails** — keep primitive opt-in, document why, do NOT promote.

**Exit decision:**
- All G1–G5 pass → promote `babel_codec` to default feature, update README, file follow-up issue #002 for the LatCal chain-commitment Super-GOAT investigation.
- G2 fails → keep opt-in, document honest negative result, do NOT promote. Match the CompressionDrafter precedent.

---

## What stays OUT of this plan

- **riir-ai integration** (NPC dialog memory, npc_comms text channel, Engram text-side compressor) — separate riir-ai plan after G2 passes.
- **riir-chain LatCal commitment of compressed KG triples** — issue #002, only after G2 + G5 pass.
- **Learned / LLM-prompted BabelCodec** (the paper's headline 3.6× number) — → riir-train if pursued. The modelless fixed-rule subset is the scope here.
- **Cross-model transfer validation on our NPC model zoo** — riir-ai scope, after the open primitive lands.

---

## File change summary

| File | Change |
|------|--------|
| `crates/katgpt-core/src/babel_codec/mod.rs` | New: trait + module docs |
| `crates/katgpt-core/src/babel_codec/fixed_rule.rs` | New: BT-P8 schema codec |
| `crates/katgpt-core/src/babel_codec/sigmoid_latent.rs` | New: latent projection codec |
| `crates/katgpt-core/src/babel_codec/commitment.rs` | New: BLAKE3 commitment |
| `crates/katgpt-core/Cargo.toml` | Add `babel_codec` feature (opt-in) |
| `crates/katgpt-core/src/lib.rs` | Add `#[cfg(feature = "babel_codec")] pub mod babel_codec;` |
| `tests/bench_331_babel_codec_goat.rs` | New: G1–G5 gate |
| `.benchmarks/331_babel_codec_goat.md` | New: results (after Phase 5) |
| `README.md` | Add BabelCodec section (after Phase 5 promotion) |

---

## References

- Research: [katgpt-rs/.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md)
- Source paper: [arxiv 2606.19857](https://arxiv.org/abs/2606.19857) — Zhu et al., SJTU, Jun 2026
- Latent-level cousin (already shipped): [riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md](../../riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md), Plan 311
- CompressionDrafter failure precedent: Plan 285, Plan 287, [`.benchmarks/285_compression_drafter_goat.md`](../.benchmarks/285_compression_drafter_goat.md)
- Compressor-reader pair analog: Plan 025 (`LoraPair { reader, writer }`)
- Super-GOAT-conditional follow-up: [`.issues/002_deterministic_babeltele_chain_commitment.md`](../.issues/002_deterministic_babeltele_chain_commitment.md)

## TL;DR

Plan 331 = `BabelCodec` trait + `FixedRuleTextCodec` (BT-P8 deterministic text codec, the modelless subset of BabelTele) + `SigmoidLatentCodec<D>` (generic-trait facade over existing `DensityBudget` infrastructure) + BLAKE3 commitment. Opt-in `babel_codec` feature until G2 (≥ 2× on real Seal 17k corpus) passes — the same gate that killed CompressionDrafter twice. Honest negative result if G2 fails. Cross-arch determinism (G5) is required to unblock issue #002 (LatCal chain commitment of compressed KG triples). riir-ai integration (NPC dialog memory, npc_comms text channel, Engram text-side compressor) is a separate plan after G2 passes.
