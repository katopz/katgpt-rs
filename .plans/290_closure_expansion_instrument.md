# Plan 290: Closure-Expansion Instrument (CEI) — PTG + Motif Mining + PRI/CDG/TaR Metrics

**Date:** 2026-06-18
**Research:** [katgpt-rs/.research/264_Compositional_Open_Ended_Intelligence_Framework.md](../.research/264_Compositional_Open_Ended_Intelligence_Framework.md)
**Source paper:** [arxiv 2606.15386](https://arxiv.org/abs/2606.15386) — Momennejad & Raileanu, "A Compositional Framework for Open-ended Intelligence", Jun 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/closure/` (new module) + Cargo feature `closure_instrument`
**Status:** Active — Phase 0 spec-locked; Phases 1–4 unstarted

---

## Goal

Ship the runtime/data-structure half of the Compositional Open-ended Intelligence paper that is *not* already in our stack. Specifically: a **Primitive Transition Graph (PTG)** recorder that wraps any `ConstraintPruner` execution; a **motif miner** that discovers recurring subgraphs across recent PTGs; and the **PRI / CDG / TaR** evaluation metrics that turn "open-ended inference" into a *measurable* property of our NPCs.

This is a measurement + data-structure layer, not a new capability class. It fuses with three already-shipped pillars (Plan 215 MDL gate, MUSE skill lifecycle, `AnchorProfile` cross-game transfer) and converts their outputs into observable metrics + promotes recurring motifs into higher-order primitives via the existing MDL admission gate.

**GOAT gate (must pass all of G1–G4 before promoting to default-on; G5 is the demotion rule):**
- G1: PRI computation < 100µs per 1K-trace corpus (Hot-tier).
- G2: Motif mining adds < 5% overhead to regime-transition admission path when feature is enabled.
- G3: TaR metric correlates ≥ 0.5 with measured cross-game transfer acceleration from `AnchorProfile` benchmarks (riir-ai).
- G4: PTG snapshot of 10K traces serializes to < 1MB for cold-tier commitment.
- G5 (demotion): if metrics don't correlate with any existing quality/transfer benchmark after Phase 4, demote `closure_instrument` to opt-in diagnostic only. Do NOT promote to default-on.

---

## Phase 0 — Spec Lock (CURRENT)

### Tasks

- [x] **T0.1** Read paper, run fusion protocol grep, write Research 264 (this session).
- [x] **T0.2** Lock PTG data structure shape (see §Data Model below).
- [x] **T0.3** Lock raw-vs-latent boundary: PTG structure = raw/syncable; motif embeddings = latent/local; TaR scalar = diagnostic (see Research 264 §4).
- [x] **T0.4** Lock GOAT gate G1–G5.
- [x] **T0.5** Lock module layout under `crates/katgpt-core/src/closure/` (no game semantics — katgpt-rs is public engine).

---

## Phase 1 — PTG Data Structure + Recorder (CORE)

The minimal "every trace becomes a directed graph" primitive. Zero-allocation recorder, lock-free `push` for plasma-tier hot path.

### Tasks

- [x] **T1.1** `mod.rs`: define `PrimitiveKind` (enum, `#[repr(u32)]`, stable enumeration for sync/replay), `OperatorKind` (`#[repr(u8)]`: `Sequence`, `Branch`, `Recurse`, `ParallelJoin`), `PtgNode { primitive: PrimitiveKind, tick: u32, blake3_in: [u8;32] }`, `PtgEdge { op: OperatorKind, from: u32, to: u32 }`, `PrimitiveTransitionGraph { nodes: Vec<PtgNode>, edges: Vec<PtgEdge>, root: u32 }`.
- [x] **T1.2** `trace.rs`: `PtgRecorder` — wraps any `ConstraintPruner` execution. Methods: `enter(primitive) -> NodeId`, `exit(node_id, op, child_id)`, `finish() -> PrimitiveTransitionGraph`. Use `smallvec::SmallVec<[PtgNode; 16]>` for typical short traces; spill to `Vec` only on overflow. **Zero allocations on the hot path when `closure_instrument` feature is disabled.**
- [x] **T1.3** `serde` impls for PTG (CBOR or postcard for cold-tier; commitment = `blake3::hash(serialized)`).
- [x] **T1.4** Property test: `PtgRecorder` output is deterministic given the same call sequence + RNG seed. Property test: serialization round-trip preserves structure.
- [x] **T1.5** Unit test: PTG of a 4-pruner operadic composition (use `lattice_operad/composed_pruner.rs` fixture) materializes to expected `(nodes, edges)`.

### Acceptance

- `PtgRecorder::enter` + `exit` combined < 200ns per call (plasma-tier) on M-series / x86_64.
- Zero allocations when feature disabled (`#[cfg(feature = "closure_instrument")]` gates every public API).

---

## Phase 2 — Motif Mining + Promotion (FUSION WITH PLAN 215)

Discover recurring subgraphs across recent PTGs and admit high-PRI motifs as new composite primitives through the existing `RegimeTransitionGate` (Plan 215). This is the paper's §4.4 "Discovering Motifs" + §5.2 "wrapped motifs become higher-order primitives".

### Tasks

- [x] **T2.1** `motif.rs`: define `Motif { subgraph_hash: [u8;32], node_count: u8, edge_count: u8, occurrence_count: u32, task_family_ids: FixedU32Set<16> }`. `task_family_ids` = which distinct task families the motif appeared in (drives PRI).
- [x] **T2.2** `motif.rs`: `MotifMiner { recent_ptgs: RingBuffer<PrimitiveTransitionGraph, K=1024>, motif_index: papaya::HashMap<[u8;32], Motif> }`. Use bounded-depth gSpan-lite algorithm (max motif size = 4 nodes / 4 edges) — O(K · 4^4) worst case at K=1024 traces ≈ 256K ops, fits comfortably in warm tier (ms).
- [x] **T2.3** `motif.rs`: `MotifMiner::mine_batch() -> Vec<Motif>` — runs in rayon at sleep-cycle boundaries (like AutoDreamer Plan 107). Does NOT run on the decode hot path.
- [x] **T2.4** `admit.rs`: `MotifAdmitter::evaluate(motif: &Motif, gate: &RegimeTransitionGate) -> GateResult`. Wraps Plan 215's gate; admission cost scales with `motif.node_count` (more nodes = higher admission cost, mirroring MDL). Accept iff `DL_new < DL_old - AdmissionCost(motif)`.
- [x] **T2.5** `admit.rs`: when a motif is admitted, register it as a new `PrimitiveKind::Composite(motif_hash)` variant. Future PTGs that match the motif emit a single composite-primitive node instead of the underlying subgraph (compression).
- [x] **T2.6** Integration test: synthesize 100 PTGs containing the same 3-node motif (e.g., `Search → Verify → Branch`). Run `mine_batch()` → motif discovered. Run `MotifAdmitter::evaluate()` → admitted. Run a 101st PTG → emits as single composite node.
- [x] **T2.7** Demotion test: a candidate motif that appears only in 1 task family (low PRI) is rejected by the gate even if `occurrence_count` is high.

### Acceptance

- `MotifMiner::mine_batch()` over 1K traces completes in < 5ms (warm-tier).
- Motif admission path adds < 5% overhead to existing `RegimeTransitionGate::evaluate()` (G2).

---

## Phase 3 — PRI / CDG / TaR Metrics (FUSION WITH ANCHORPROFILE)

The paper's §6 evaluation metrics as runtime functions. PRI/CDG are pure-PTG-aggregate; TaR requires the `AnchorProfile` cross-game transfer machinery (already in riir-ai as private IP — this plan exposes only the *metric*, not the transfer mechanism).

### Tasks

- [x] **T3.1** `metrics.rs`: `PrimitiveReuseIndex::compute(corpus: &[PrimitiveTransitionGraph]) -> HashMap<PrimitiveKind, f32>`. PRI(p) = (count of distinct task families containing p) / (total task families). Stored per-primitive.
- [x] **T3.2** `metrics.rs`: `CompositionalDepthGeneralization::compute(train_depths: &[u32], test_depth: u32, success_rate: f32) -> CdgScore`. Returns rolling EMA of "success rate at depth > max training depth seen". Stored as scalar per-NPC.
- [x] **T3.3** `metrics.rs`: `TransferAsRecomposition::compute(baseline_ptgs: &[PTG], perturbed_ptgs: &[PTG], anchor_profile: &AnchorProfileRef) -> f32`. **Latent-only output.** Compares motif distributions before/after environment perturbation. Output ∈ [0, 1]; high = same motifs still solve perturbed instances (good TaR).
- [x] **T3.4** Bridge function `ptg_to_motif_embedding(ptg: &PTG, motif_directions: &MotifDirections) -> [f32; K]` (raw→latent). Zero-allocation dot-product projection + sigmoid (per AGENTS.md: never softmax). `MotifDirections` = pre-computed `[f32; K*N]` lookup table loaded once per NPC personality.
- [x] **T3.5** Bridge function `motif_embedding_to_tar_score(emb: &[f32; K]) -> f32` (latent→raw scalar). Clamp to [0, 1].
- [x] **T3.6** Unit tests: PRI computation correctness on synthetic corpus (3 task families, 5 primitives). TaR computation: 100% same motifs → TaR = 1.0; 0% overlap → TaR = 0.0.

### Acceptance

- PRI computation on 1K-trace corpus < 100µs (G1).
- TaR computation on 2×100-PTG corpus < 1ms.
- All bridge functions are `#[inline]`, zero-alloc, and SIMD-friendly (use `simd_dot_f32` from `katgpt-core/src/simd.rs`).

---

## Phase 4 — GOAT Gate + Integration

### Tasks

- [x] **T4.1** Create `tests/bench_290_closure_instrument_goat.rs` with all G1–G5 assertions.
- [ ] **T4.2** Wire `PtgRecorder` as an opt-in wrapper around `BanditPruner` / `AbsorbCompressLayer` (gated by `closure_instrument` feature).
- [ ] **T4.3** Wire `MotifMiner::mine_batch()` into the existing sleep-cycle scheduler (Plan 107 AutoDreamer consolidation tick). Document the schedule.
- [ ] **T4.4** Cross-repo validation: request riir-ai to expose `AnchorProfile::translate_priorities()` benchmark traces for TaR correlation (G3). If riir-ai cannot share traces (private IP), use synthetic transfer scenarios as a proxy and downgrade G3 to "correlates with synthetic transfer" with a TODO to upgrade.
- [x] **T4.5** Cold-tier commitment: PTG snapshot ≤ 1MB per 10K traces (G4). Use postcard encoding + BLAKE3 hash; reuse Plan 280 Merkle-octree commitment infrastructure. (Honest result: 1.774 MB — see `.benchmarks/290_closure_instrument_goat.md`. Per-node 32-byte `blake3_in` is structural.)
- [x] **T4.6** Run full benchmark suite with `--features closure_instrument`. Document results in `katgpt-rs/.benchmarks/290_closure_instrument_goat.md`.
- [ ] **T4.7** If G1–G4 PASS and G5 does not fire → promote `closure_instrument` to default-on, demote any loser. (G1 PARTIAL 4507µs vs 100µs target; G4 PARTIAL 1.774MB vs 1MB — not promoted.)
- [x] **T4.8** If G5 fires (metrics don't correlate) → keep opt-in, document honest negative result in benchmark file. (Opt-in retained. Honest partial result documented in `.benchmarks/290_closure_instrument_goat.md`.)

### Acceptance

- All G1–G4 pass with measurable numbers in `.benchmarks/290_*.md`.
- README feature-showcase section added under `🔀 Opt-In & Gated Features` (or `## 🚀 Key Results` if promoted to default).
- Documentation index updated.

---

## Data Model (locked in Phase 0)

```rust
// crates/katgpt-core/src/closure/mod.rs

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PrimitiveKind {
    // 0..=255 — open primitive enumeration space (katgpt-rs engine)
    UserDefined(u32) = 0,
    // 256..=511 — composite primitives admitted by MotifAdmitter
    Composite(/* blake3 prefix */ u32) = 256,
    // 512..   — game/runtime extensions stay in riir-ai (private)
    //         — never leak game semantics into this enum
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OperatorKind {
    Sequence = 0,
    Branch   = 1,
    Recurse  = 2,
    ParallelJoin = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct PtgNode {
    pub primitive: PrimitiveKind,
    pub tick: u32,
    pub blake3_in: [u8; 32], // commitment of input state at this node
}

#[derive(Clone, Copy, Debug)]
pub struct PtgEdge {
    pub op: OperatorKind,
    pub from: u32,
    pub to: u32,
}

#[derive(Clone, Debug)]
pub struct PrimitiveTransitionGraph {
    pub nodes: Vec<PtgNode>,
    pub edges: Vec<PtgEdge>,
    pub root: u32,
    pub task_family_id: u32,
}
```

---

## Latent vs Raw Boundary (locked in Phase 0)

| Field | Space | Synced? | Reason |
|-------|-------|---------|--------|
| `PtgNode.primitive` (enum tag) | Raw | YES | Deterministic replay needs exact primitive enumeration |
| `PtgNode.tick` | Raw | YES | Deterministic ordering |
| `PtgNode.blake3_in` | Raw (32-byte commitment) | YES (audit) | Tamper-evident |
| `PtgEdge.op` | Raw | YES | Same |
| `PrimitiveTransitionGraph` (structure) | Raw | YES (if committed) | Bit-identical replay |
| `Motif.subgraph_hash` | Raw (32-byte hash) | YES (audit) | Commitment |
| `Motif.occurrence_count` | Raw (counter) | YES (aggregate) | Deterministic from execution history |
| `Motif.motif_embedding` | Latent | NO — local | Used for similarity/clustering; not for state reconstruction |
| `PrimitiveReuseIndex` output map | Raw (counters) | YES (aggregate) | Diagnostic only |
| `TaR_score: f32` | Latent (statistical) | NO — local diagnostic | Statistical, not bit-reproducible |

**Bridge functions:**
- `ptg_to_motif_embedding()` (raw→latent): dot-product projection + sigmoid. Zero-alloc, SIMD-friendly.
- `motif_embedding_to_tar_score()` (latent→raw scalar): clamp to [0, 1].

**Anti-pattern (per AGENTS.md):** never sync the `motif_embedding` vector. Only the scalar `TaR_score` crosses the sync boundary, and only as a diagnostic (NOT for anti-cheat validation — TaR is not a movement claim).

---

## Fusion Connections (Force Multiplier)

| Existing Pillar | Connection |
|-----------------|------------|
| **Plan 215 Regime-Transition + MDL gate** | `MotifAdmitter` wraps `RegimeTransitionGate::evaluate()`. Every motif admission is a vocabulary-change event in the existing regime-transition provenance chain. |
| **MUSE skill lifecycle (Plan 172/192)** | Adds `promote_motif` as a 6th lifecycle action alongside explore/patch/split/compress/retire. Driven by PRI threshold. |
| **Bayesian posterior skill evolution (R211)** | `Motif.motif_embedding` participates in BAKE precision-gated evolution — motifs with low precision (high uncertainty across worlds) get explored; high-precision motifs get promoted. |
| **`AnchorProfile` cross-game transfer (riir-ai private)** | TaR metric consumes `AnchorProfile.translate_priorities()` outputs to score how well motifs survive environment perturbation. The metric is public (katgpt-rs); the transfer mechanism is private (riir-ai). |
| **CGSP self-play (Plan 274/282)** | Every CGSP cycle's solver output gets a PTG. PRI over the CGSP trace corpus identifies which primitives generalize across game families vs which are game-specific. |
| **AutoDreamer sleep-cycle (Plan 107/116)** | `MotifMiner::mine_batch()` runs at sleep-cycle boundaries, alongside existing memory consolidation. |
| **EventLog game-trace fork-diff (Plan 124)** | PTG generalizes EventLog — same fork-diff primitives apply, plus graph structure. Long-term: PTG supersedes EventLog for new arenas. |
| **Merkle-octree cold-tier commitment (Plan 280)** | PTG snapshots serialize via postcard → BLAKE3 → Merkle-octree. Same commitment infrastructure. |

---

## What This Plan Does NOT Do

- **No NPP training objective.** That is riir-train territory. This plan exposes a *runtime data structure* (`PTG`) that a future NPP trainer could consume as training targets, but does not implement the trainer.
- **No new capability class.** This is a measurement + data-structure layer. The capability-class mechanisms (MDL admission, TaR transfer) already ship; we are making them observable.
- **No riir-ai guide.** Verdict is GOAT, not Super-GOAT. The private selling-point doc for `AnchorProfile` already exists in `cgsp_runtime/cross_game_transfer.rs` doc comments. This plan does not duplicate or extend that.
- **No changes to existing default-on features.** Everything is behind `closure_instrument` feature flag until G1–G4 pass.
- **No game semantics in katgpt-rs.** `PrimitiveKind` reserves 0–511 for engine use; game-specific primitive IDs stay in riir-ai and reference back via opaque `u32`.

---

## TL;DR

Plan 290 ships the runtime half of Momennejad & Raileanu's Compositional Open-ended Intelligence paper that isn't already in our stack: a **Primitive Transition Graph (PTG)** recorder, a **motif miner** that promotes recurring subgraphs as new primitives via the existing MDL gate (Plan 215), and **PRI/CDG/TaR metrics** that convert open-ended inference from a claim into a measured property of our NPCs. Fuses with 7 existing pillars (Plan 215 MDL gate × MUSE lifecycle × Bayesian posterior × AnchorProfile transfer × CGSP × AutoDreamer × Merkle-octree). Behind `closure_instrument` feature flag; promote to default-on only if G1–G4 PASS and G5 (correlation-with-real-quality) does not fire. **GOAT, not Super-GOAT** — measurement layer, not new capability class.
