# Issue 002: Deterministic BabelTele → LatCal Chain Commitment of KG Triples (Super-GOAT-conditional investigation)

**Filed:** 2026-06-26
**From:** [katgpt-rs/.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md) §2.4 (Fusion TBD)
**Blocks:** Super-GOAT promotion of BabelTele fusion (currently GOAT in Research 312)
**Depends on:** [Plan 331](../.plans/331_babel_codec_readability_relaxed_semantic_codec.md) Phase 5 GOAT gate (G2 + G5 must pass first)

---

## The question

Can a **deterministic** BabelTele fixed-rule codec (BT-P8 schema from paper Appendix C.2.8) compress semantic KG triples enough to make LatCal chain commitment of those triples net-positive on byte cost + commitment gas, while preserving bit-identical replay determinism?

If YES → the BabelTele × LatCal fusion is a Super-GOAT (novel sync-boundary bridge: cheaper chain commitment of semantic content). File a riir-chain/.research/ guide + riir-chain/.plans/ plan.

If NO → BabelTele stays GOAT (Research 312 verdict unchanged). Close this issue with the negative result.

---

## Why this is Super-GOAT-conditional (not committed in Research 312)

Research 312 verdict'd BabelTele as **GOAT, not Super-GOAT** because Q2 ("new class of behavior?") is uncertain — the latent-level adaptive-bandwidth NPC comms already ships (`npc_comms` Plan 311 via `NpcLatentMessage { hla_slice }` + `DensityBudget`), so the text-level codec is incremental on an existing capability class.

The one path that could flip Q2 to YES is the **sync-boundary bridge**: if deterministic BT-P8 compressed KG triples can be LatCal-committed and replayed bit-identically across architectures, that creates a genuinely new capability class — "compressed semantic content crosses the chain sync boundary at 3–4× lower byte cost without breaking deterministic replay". No shipped primitive does this.

Per the research skill rule ("do not write 'Super-GOAT candidate' as a deferred-commitment escape hatch"), this fusion is **not** claimed as Super-GOAT in Research 312 — it is tracked here as an open investigation.

---

## The three gates (all must pass for Super-GOAT promotion)

| Gate | Metric | Pass if | Analog |
|------|--------|---------|--------|
| **BG1** (fidelity) | Round-trip `decompress(compress(triple)) ≡ triple` on 1000 real KG triples from our KG emission paths | 100% bit-identical | Plan 331 G1 |
| **BG2** (compression) | Byte reduction on the same 1000 triples | ≥ 2× (honest target: 3×) | Plan 331 G2 |
| **BG3** (determinism) | Same triple → bit-identical compressed bytes + BLAKE3 commitment across ARM64 / x86_64 / wasm32 | 0 mismatches / 1000 / 3 archs | Plan 331 G5, riir-chain CG1 |
| **BG4** (commitment cost) | LatCal commitment cost of compressed triple vs uncompressed | ≤ 0.5× uncompressed cost (byte savings must exceed commitment overhead) | riir-chain CG2 |
| **BG5** (replay) | Cold-tier replay reconstructs original triple from compressed form | 100% success on 1000 triples | riir-chain catchup tests |

---

## Pre-conditions (must land first)

1. **Plan 331 Phase 5** — `FixedRuleTextCodec` ships, G1 + G2 + G5 pass on real Seal corpus. If Plan 331 G2 fails, this issue closes as moot (no codec to commit).
2. **riir-chain `KgSpectralPayload` or equivalent triple-commitment path** identified — need a concrete commit-and-replay surface to measure BG4/BG5 against.

---

## Investigation steps (after pre-conditions land)

- [ ] **I1** Pull 1000 real KG triples from our KG emission paths (`riir-ai/crates/riir-engine/src/kg*.rs`, `riir-neuron-db/src/vibe.rs`).
- [ ] **I2** Run Plan 331 `FixedRuleTextCodec` on the 1000 triples → measure BG1 (round-trip), BG2 (compression).
- [ ] **I3** Run Plan 331 G5 cross-arch determinism on the same 1000 triples → measure BG3.
- [ ] **I4** Wire `FixedRuleTextCodec` output into the riir-chain LatCal commitment path → measure BG4 (commitment cost vs uncompressed).
- [ ] **I5** Run cold-tier replay → measure BG5.
- [ ] **I6** Verdict:
  - All BG1–BG5 pass → file `riir-chain/.research/NNN_Deterministic_BabelTele_LatCal_KG_Commitment_Guide.md` (Super-GOAT guide), update Research 312 verdict GOAT → Super-GOAT (cross-ref), file riir-chain plan.
  - Any fail → close this issue with the negative result. BabelTele stays GOAT.

---

## Honest expectations

- **BG1 (round-trip):** likely PASSES — BT-P8 schema is deterministic and well-defined for `(subject, predicate, object)` triples.
- **BG2 (compression):** likely PASSES — KG triples are highly compressible (`(Wang_Nianfang, appellant_of, Hubei_Longan_Real_Estate)` → `*(Wang_Nianfang):appellant_of=Hubei_Longan_Real_Estate` is comparable length; the win is on attribute lists and multi-predicate subjects).
- **BG3 (cross-arch determinism):** likely PASSES — fixed-rule mapping is architecture-independent by construction. BLAKE3 is cross-platform.
- **BG4 (commitment cost):** UNCERTAIN — LatCal commitment cost may not scale linearly with byte count (matrix operations dominate). The byte savings might be eaten by fixed commitment overhead. This is the load-bearing gate.
- **BG5 (replay):** likely PASSES if BG1 passes — replay is just decompress-after-commit.

**Predicted outcome:** BG4 is the likely failure point. If LatCal commitment has fixed overhead per payload (independent of byte count), then BabelTele compression saves bytes but not commitment cost → no net win → stays GOAT. If commitment cost scales with byte count, then 3× byte savings → 3× commitment savings → Super-GOAT.

---

## Cross-references

- Source research: [katgpt-rs/.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md) §2.4
- Codec implementation: [katgpt-rs/.plans/331_babel_codec_readability_relaxed_semantic_codec.md](../.plans/331_babel_codec_readability_relaxed_semantic_codec.md)
- riir-chain LatCal: `riir-chain/src/encoding/latcal*.rs`, `riir-chain/.research/004_LatCal_Fixed_Point_Bridge_Lean4_Proof_Guide.md`
- riir-chain commitment gate pattern: `riir-chain/.research/001_Resolution_Tiered_Deterministic_Commitment_Guide.md` §CG1–CG6
- Raw-sync rule reminder (AGENTS.md): KG triples are semantic → can be compressed; raw position/HP/wallet MUST stay uncompressed.

## TL;DR

Open investigation: can deterministic BT-P8 BabelTele compression of KG triples survive LatCal chain commitment + cold-tier replay with net byte savings? If yes (all BG1–BG5 pass), BabelTele × LatCal is a Super-GOAT (file riir-chain guide). If no (likely BG4 fails on commitment-cost scaling), BabelTele stays GOAT (Research 312 unchanged). Blocked on Plan 331 Phase 5 GOAT gate landing first. Predicted bottleneck: BG4 commitment cost scaling — the load-bearing gate.
