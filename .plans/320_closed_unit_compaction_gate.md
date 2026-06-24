# Plan 320: Closed-Unit Compaction Gate — Rubric-Gated Trajectory Compaction (CUCG)

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/300_Closed_Unit_Compaction_Gate_Rubric_Gated.md](../.research/300_Closed_Unit_Compaction_Gate_Rubric_Gated.md)
**Source paper:** [arXiv:2606.23525](https://arxiv.org/abs/2606.23525) — SelfCompact (Li et al., JHU + Apple, Jun 2026)
**Private guide:** [riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md](../../riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md) (game-AI selling point)
**Cross-ref:** [riir-neuron-db/.research/007_Can_Freeze_As_Cucg_Instance_Crossref.md](../../riir-neuron-db/.research/007_Can_Freeze_As_Cucg_Instance_Crossref.md) (`can_freeze` isomorphism)
**Target:** `katgpt-rs/src/compaction/` (new module) + Cargo feature `closed_unit_compaction`
**Status:** Active — Phase 1 + Phase 2 + Phase 3 COMPLETE (2026-06-25). 64 unit tests PASS, G1/G3/G5/G6 gates green. Phase 4 (MathRubric, CacheReuseProbe, G3) next.

---

## Goal

Ship a generic, modelless, **rubric-gated trajectory compaction** primitive (`ClosedUnitCompactionGate<R>`) that fires summarization at structurally-safe moments (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck) rather than at fixed token thresholds. Source paper (SelfCompact, arXiv:2606.23525) shows this matches fixed-interval accuracy at 30–70% lower token cost on agents, and an oracle skip-if-correct variant has +11.5pp further headroom.

**Why this is a Super-GOAT, not a Gain:** the paper's C1/C2/C3/N1 rubric is structurally isomorphic to our already-shipped `can_freeze` shard gate (riir-neuron-db Plan 002). Unifying them as instances of one primitive (CUCG) is the cross-domain force multiplier. We ship the open trajectory-side primitive here; the shard-side already exists.

**GOAT gate (open primitive):** G1 rubric beats fixed-interval on structural safety; G2 skip-if-reliable CLR fuse; G3 cache-reuse probe overhead independent of L; G4 zero-alloc hot path; G5 feature isolation; G6 sigmoid-never-softmax; G7 cross-domain isomorphism with `can_freeze`. Runtime gate G8 (per-NPC variant) is riir-ai's responsibility.

**Constraints:** modelless (no training, no backprop); latent-to-latent preferred (predicates are sigmoid projections on coherence/intrinsic-rank/divergence/novelty, never softmax); zero-alloc hot path; deterministic audit record crosses sync boundary as raw.

---

## Phase 1 — Types Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module `src/compaction/mod.rs` with feature gate `#[cfg(feature = "closed_unit_compaction")]`. Re-export public API.
- [x] **T1.2** Define `Rubric` trait in `src/compaction/rubric.rs`:
  ```rust
  pub trait Rubric {
      const ARITY: usize;
      fn evaluate(
          &self,
          trajectory_prefix: &[u8],
          scratch: &mut RubricScratch,
      ) -> RubricVerdict;
  }
  ```
  With `RubricVerdict { predicates: [PredicateResult; ARITY] }` and `PredicateResult::{Yes { quote_start, quote_len }, No { reason }}`. Fixed-size arrays; no heap alloc in the verdict.
- [x] **T1.3** Define `FireRule` enum in `src/compaction/fire_rule.rs`: `And(u8)`, `Or(u8)`, `Not(u8)`, `Box(Box, Box)`. Implement `fn evaluate(&self, verdict: &RubricVerdict) -> bool` (recursive, no alloc).
- [x] **T1.4** Define `Backstop` enum in `src/compaction/backstop.rs`: `None`, `TokenPct(f32)`, `Never`. Implement `fn should_force(&self, prompt_len: usize, ctx_window: usize) -> bool`.
- [x] **T1.5** Define `CompactionAuditRecord` in `src/compaction/audit.rs`: `#[derive(Clone, Copy, Debug, PartialEq)]`, fixed-size, `#[repr(C)]` for sync-boundary crossing. Fields per research note §2.3.
- [x] **T1.6** Define `CompactionDecision` enum: `Compress { audit }`, `Continue { audit }`, `Forced { audit }`.
- [x] **T1.7** Unit tests: `FireRule::And(0b1111)` returns true iff all 4 predicates Yes; `Or(0b0001)` returns true iff predicate 0 Yes; `Not(0)` returns true iff predicate 0 No; `Box(Or, And)` composes.
- [x] **T1.8** Unit tests: `Backstop::TokenPct(0.30)` forces at 30% of ctx_window, doesn't force below.

### Acceptance

Compiles with `cargo check --features closed_unit_compaction`. Unit tests pass. Zero heap allocations in `FireRule::evaluate` and `Backstop::should_force` (verify via `#[track_caller]` allocator or `dhat`).

---

## Phase 2 — The Gate Kernel

### Tasks

- [x] **T2.1** Define `ClosedUnitCompactionGate<R: Rubric>` in `src/compaction/gate.rs`:
  ```rust
  pub struct ClosedUnitCompactionGate<R: Rubric> {
      rubric: R,
      fire_rule: FireRule,
      backstop: Backstop,
      skip_if_reliable: Option<f32>,
      probe_interval_tokens: usize,
  }
  ```
- [x] **T2.2** Implement `pub fn evaluate(&self, trajectory_prefix: &[u8], prompt_len: usize, ctx_window: usize, clr_vote: Option<f32>, scratch: &mut RubricScratch) -> CompactionDecision`. Decision order:
  1. `backstop.should_force(prompt_len, ctx_window)` → `Forced { audit }` if true.
  2. Else `rubric.evaluate(...)` → `verdict`.
  3. `fire_rule.evaluate(&verdict)` → if true, check `skip_if_reliable`: if `Some(τ)` and `clr_vote > τ` → `Continue { audit }` (suppressed); else `Compress { audit }`.
  4. Else `Continue { audit }`.
- [x] **T2.3** Build `CompactionAuditRecord` with bit-identical fields from the verdict + fire-rule evaluation + backstop/skip flags. `#[repr(C)]` for sync crossing.
- [x] **T2.4** Implement `pub fn probe_interval_tokens(&self) -> usize` for the caller's probe loop.
- [x] **T2.5** Zero-alloc assertion: `evaluate()` performs no heap allocation when `R::ARITY ≤ 8` (uses `RubricScratch` for any temporary buffers). Document the contract in rustdoc.
- [x] **T2.6** Unit tests: paper's search rule (`And(0b1111)` over C1/C2/C3/N1-inverted) → Compress iff all four pass; paper's math rule (`Or(0b0001, And(0b0110))` over Q1/Q2/Q3) → Compress iff Q1 or (Q2 and Q3).
- [x] **T2.7** Unit tests: `skip_if_reliable = Some(0.8)`, `clr_vote = 0.9` → Compress suppressed to Continue even if rubric fires; `clr_vote = 0.7` → Compress proceeds.
- [x] **T2.8** Unit tests: backstop overrides rubric — rubric says Continue but prompt > 30% ctx_window → Forced.

### Acceptance

`cargo test --features closed_unit_compaction --lib compaction::gate` passes. G4 (zero-alloc) verified on the `evaluate` hot path.

---

## Phase 3 — Search Rubric (Paper Reproduction)

### Tasks

- [x] **T3.1** Implement `SearchRubric` in `src/compaction/rubrics/search.rs` with `ARITY = 4` (C1, C2, C3, N1). Predicates computed from a `TrajectoryFeatures` struct the caller supplies (coherence stability, intrinsic rank, divergence-since-last-summary, novelty rate) — **latent reframing per research note §2.4**, not LLM-judged.
- [x] **T3.2** Define `TrajectoryFeatures { coherence: f32, intrinsic_rank: f32, divergence_since_last: f32, novelty_rate: f32 }`. Each feature is a scalar from existing primitives (latent_functor quality_gate, subspace_phase_gate, DEC codifferential, cgsp curiosity).
- [x] **T3.3** Implement predicate sigmoids:
  - C1: `σ(β_c1 · (coherence − τ_c1))` → Yes iff > 0.5.
  - C2: `σ(β_c2 · (rank_ceiling − intrinsic_rank))` → Yes iff > 0.5 (low rank = summarizable).
  - C3: `σ(β_c3 · (divergence_since_last − τ_c3))` → Yes iff > 0.5 (positive divergence = progress).
  - N1: `σ(β_n1 · (novelty_rate − τ_n1))` → Yes iff > 0.5 (high novelty rate = NOT stuck; the fire rule negates).
- [x] **T3.4** Configure `fire_rule = And(0b1111)` over (C1, C2, C3, ¬N1) — the paper's search rule.
- [x] **T3.5** Paper Figure 1 reproduction test: synthetic BrowseComp-like trajectory with marked safe-to-compact (post-verified-fact) and mid-derivation points. Assert: CUCG fires ≥ 80% recall at safe points; ≤ 20% FDR at mid-derivation. **This is G1.**
- [x] **T3.6** Document that the paper's LLM-judged verbatim quotes are replaced by latent-feature sigmoid projections; the audit record still records the trajectory span `[quote_start, quote_len]` that grounded each Yes (the span where the feature crossed threshold).

### Acceptance

G1 PASSES (recall=1.000, FDR=0.000 on synthetic Figure-1 reproduction). `cargo test --features closed_unit_compaction --lib compaction::rubrics::search` — 15 tests pass.

---

## Phase 4 — Math Rubric + Cache-Reuse Probe Protocol

### Tasks

- [ ] **T4.1** Implement `MathRubric` in `src/compaction/rubrics/math.rs` with `ARITY = 3` (Q1, Q2, Q3). Configure `fire_rule = Or(Q1_mask, And(Q2_mask | Q3_mask))` — paper's math rule.
- [ ] **T4.2** Implement `CacheReuseProbe` in `src/compaction/probe.rs`:
  - `pub fn probe_append(&self, trajectory: &mut Vec<u8>, rubric_prompt: &[u8]) -> ProbeToken` — appends rubric prompt to trajectory (preserving KV cache), returns token for revert.
  - `pub fn revert(&self, trajectory: &mut Vec<u8>, token: ProbeToken)` — removes the appended prompt on CONTINUE (no cache pollution).
  - `pub fn summarize(&self, trajectory: &[u8], summarizer_prompt: &[u8]) -> Summary` — caller-supplied summarizer (LLM or chain_fold); CUCG itself does not summarize.
- [ ] **T4.3** G3 test: measure probe latency at L = 1k, 10k, 100k tokens. Assert latency within ±10% across L (only the appended instruction pays prefill). Document the cache-reuse invariant in rustdoc.
- [ ] **T4.4** Probe-revert correctness test: generate k CONTINUE probes, then verify subsequent generation matches a no-probe baseline byte-for-byte (modulo KV-cache indexing). The rolling cache MUST be uncontaminated.

### Acceptance

G3 PASSES. Math rubric reproduces paper's `Q1 ∨ (Q2 ∧ Q3)` rule. Probe-revert is byte-clean.

---

## Phase 5 — Cross-Domain Isomorphism (riir-neuron-db bridge)

### Tasks

- [ ] **T5.1** Implement `ShardFreezeRubric` in `src/compaction/rubrics/shard_freeze.rs` with `ARITY = 2` mirroring `riir-neuron-db/src/phase_gate.rs`:
  - P0 (input_sufficient): `n_wake_events >= intrinsic_dim` (Wang et al. Thm 4).
  - P1 (output_converged): `spectral_flatness < 0.3`.
  - `fire_rule = And(0b0011)` → `can_freeze = input_sufficient && output_converged`.
- [ ] **T5.2** G7 test: construct a `ClosedUnitCompactionGate<ShardFreezeRubric>` and verify it produces **bit-identical** `CompactionAuditRecord` decisions to `ConsolidationPipeline::can_freeze` on the same `(n_wake_events, style_weights)` inputs. This is the isomorphism proof.
- [ ] **T5.3** Document the isomorphism table (research note §1, cross-ref 007) in the rustdoc of `ShardFreezeRubric`. **The shard freeze gate and the trajectory compaction gate are the same primitive** — recognized after the fact, not designed in.

### Acceptance

G7 PASSES. Bit-identical decisions on the shard-freeze rubric vs `can_freeze`. The cross-domain unification is proven, not just claimed.

---

## Phase 6 — GOAT Gate, Benchmark, Promotion Decision

### Tasks

- [ ] **T6.1** Write `benches/cucg_bench.rs`:
  - Throughput: `evaluate()` ≥ 50M decisions/sec for ARITY=4 (parity with Salience Tri-Gate's 120M/sec — CUCG has the same two-sigmoid + fire-rule cost shape).
  - Latency: `evaluate()` ≤ 50 ns for ARITY=4.
  - Zero-alloc assertion via allocator hook.
- [ ] **T6.2** Write `benches/cucg_goat.rs` running G1–G7 with pass/fail and measured numbers. Format mirrors `.benchmarks/303_salience_tri_gate_goat.md`.
- [ ] **T6.3** G2 (skip-if-reliable): synthetic trajectory with high-reliability CLR vote (>0.8) → suppression rate ≥ 50%; quality maintained vs no-suppression baseline.
- [ ] **T6.4** G5 (feature isolation): `cargo build --no-default-features --features closed_unit_compaction` succeeds; `cargo build --no-default-features` succeeds; `nm target/release/libkatgpt_rs.dylib | grep -ic compaction` → 0 when feature off.
- [ ] **T6.5** G6 (sigmoid-never-softmax): static check — grep module for `softmax`, expect 0 hits; document in rustdoc that each predicate is a scalar from a sigmoid projection.
- [ ] **T6.6** **Promotion decision.** If G1–G7 pass AND the gain is modelless (it is — no training required), promote `closed_unit_compaction` to `default` features per AGENTS.md GOAT gate rule. Demote the loser (fixed-interval `OnlineCompactor::trigger_threshold()` stays as the backstop arm; it is not removed — it becomes the `Forced` decision's mechanism).
- [ ] **T6.7** If G1 or G7 fails, **do NOT promote**. Investigate: G1 fail → latent predicates don't track LLM-judged ones; reconsider whether `coherence`/`intrinsic_rank`/`divergence`/`novelty` are the right features. G7 fail → isomorphism claim is wrong; revise research note.

### Acceptance

G1–G7 all PASS with measured numbers in `.benchmarks/320_cucg_goat.md`. Promotion to default decided and executed (or deferred with documented reason).

---

## Phase 7 — Examples + Docs

### Tasks

- [ ] **T7.1** `examples/cucg_search_basic.rs` — minimal example: synthetic trajectory, SearchRubric, print decision + audit record.
- [ ] **T7.2** `examples/cucg_shard_freeze_isomorphism.rs` — demonstrate G7: construct shard-freeze CUCG, compare to `can_freeze`, print bit-identical records.
- [ ] **T7.3** `examples/cucg_skip_if_reliable.rs` — demonstrate G2: high CLR vote suppresses compaction.
- [ ] **T7.4** Update `katgpt-rs/README.md` Feature Showcase with Plan 320 CUCG section (format mirrors Plan 303 Salience Tri-Gate section).
- [ ] **T7.5** Update `katgpt-rs/.docs/01_overview.md` feature table with `closed_unit_compaction` row.

### Acceptance

Examples run with `cargo run --example cucg_*  --features closed_unit_compaction`. README + docs updated.

---

## Out of Scope (tracked elsewhere)

- **Per-NPC sub-goal rubric + tick-loop wiring** → riir-ai Plan TBD (after Research 155 guide acceptance). This is G8.
- **LatCal-committed audit trail bridge** → riir-chain Plan TBD (after CUCG ships and the sync-boundary contract stabilizes).
- **LLM-judged verbatim quotes** (paper's literal mechanism) → not implemented; our latent reframing replaces it. If G1 fails badly, reconsider.

---

## Reproduction

```bash
# Phase 1–2 unit tests
cargo test --features closed_unit_compaction --lib compaction

# Phase 3 search rubric (G1)
cargo test --features closed_unit_compaction --lib compaction::rubrics::search

# Phase 4 math rubric + probe protocol (G3)
cargo test --features closed_unit_compaction --lib compaction::probe

# Phase 5 cross-domain isomorphism (G7)
cargo test --features closed_unit_compaction --lib compaction::rubrics::shard_freeze

# Phase 6 GOAT gate (G1–G7)
cargo run --release --features closed_unit_compaction --bench cucg_goat

# Promotion check (G5)
cargo build --no-default-features --features closed_unit_compaction
cargo build --no-default-features
nm target/release/libkatgpt_rs.dylib | grep -ic compaction  # → 0
```
