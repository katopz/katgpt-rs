# Issue 974 — the walk-proximity context-ranker primitive (lift from riir-neuron-db issue 640)

**Status:** CLOSED 2026-10-30 — T1–T4 DONE in one arc ([Bench 974](.benchmarks/974_issue974_walk_rank_goat.md): G2 ≈ 0.29 ms/call worst on the synthetic view, 36× under bar; G1 asserted on every timed iteration; G4 alloc gate green). The ndb re-export flip landed the same day (ndb `walk_rank` forwards `katgpt-core/walk_rank`; full ndb suite + bench re-ran green through the shim). ⚠ Sequencing: ndb's git dep floats at katgpt-rs `main` — the sibling-checkout [patch] compiles against the working tree; a bare main fails cargo resolution LOUD until the next develop → main promote (recorded in ndb's Cargo.toml feature comment).

## What exists (ndb `src/experience_graph/walk_rank.rs`, feature `walk_rank`, opt-in)

- `WalkGraphView` — the ONLY graph access: `neighbors_of(hash, out)` + `tick_of(hash) -> Option<u64>`. The trait is the entire dependency surface; the math never sees a store.
- Stage-1 `walk_rank(view, query, cfg) -> Vec<WalkRankedHit>`: W non-backtracking random walks of length K from a query node; rank visited nodes by visit count desc → tick desc → hash asc (total order — byte-identical re-runs). Temporal as-of guard + dangling hashes filtered BEFORE sampling (an inadmissible candidate never perturbs the choice set — provable future-insert invariance). Per-walk rng seeds: `BLAKE3(seed ‖ walk_index)` → splitmix64 → xorshift64. Scratch allocated ONCE per call.
- Stage-2 `budgeted_context(view, ranked, query, budget)`: ranked seeds join the context then seed a bounded BFS; set accounting (each node once); the query never appears in its own context.
- Defaults W=32, K=8 (the proven headline posture).

## Evidence (why this clears the lift bar)

- **Mechanism** (ndb bench 516, planted-community toy graph): walk 0.965 p@10 / 0.891 nDCG vs cosine 0.492/0.375 (+0.47); strictly beats BFS; beats uniform; **transfers to the sparse-typed arm** (0.952 vs 0.394 — structure carries MORE signal when embeddings are weak).
- **Latency** (ndb bench 517, release): defaults 0.07–0.20 ms/call across N ∈ {1k,10k,100k} × {Chain,Branching} — FLAT in N (visits W×K nodes regardless of graph size), 50–147× under the 5–10 ms offline budget. Sensitivity: (16,8) 3.4× cheaper at −0.013 p@10.
- **Gates**: determinism (byte-identical re-runs), temporal leak-free + future-insert invariance, budget exact/capped, dangling refusal — green at every measured posture.
- Source: arXiv:2610.17867 §4.2 (two-stage retrieval) + Appendix I (Algorithm 1) — the ranking BY visit count is the load-bearing part (re-scoring by cosine collapses it to pool-widening; measured ≡ flat cosine in T1).

## Scope

- [x] T1. Move the math (trait + stage-1 + stage-2 + the rng/seed helpers) into `katgpt-core` behind opt-in `walk_rank` (no new deps — blake3 is already in-tree); ndb's `walk_rank` feature re-exports at the same paths (the Plan 319 consumer shape), feature implied by ndb's own flag.
  **DONE** (`crates/katgpt-core/src/walk_rank.rs` + feature `walk_rank = []` + lib.rs cfg-gated module/re-exports; ndb side: `src/experience_graph/walk_rank.rs` is now the shim — `pub use katgpt_core::walk_rank::*` at the same paths + the ndb-only `ExperienceGraph` methods — and ndb's `walk_rank` forwards `katgpt-core/walk_rank`). Adapters (`GraphWalkView`, `KgTripleWalkView`, `RetrievalMode::WalkProximity`) stayed consumer-side untouched.
- [x] T2. Port the 4 in-module tests + the KG/NS adapter test pattern (adapters stay CONSUMER-side — the trait means the lift moves no graph code).
  **DONE** — 6 in-module tests upstream (the ndb 4, minus the graph fixture, plus the alloc gate + the hub-dominance mechanism test over a synthetic `ToyView` with BIDIRECTIONAL link semantics); the ndb fixture tests stayed ndb-side as the adapter integration suite (they now double as the re-export wiring proof — 8 walk tests green through the shim, 512 lib tests total).
- [x] T3. GOAT: G1 determinism/leak suite + G2 latency surface re-run katgpt-core-side (bars: byte-identical re-runs; defaults ≤ 10 ms at 100k nodes — ndb 517 measured 0.13 ms worst cell, so the bar has 75× headroom); G4 alloc: once-per-call scratch, zero per-step alloc (assert with the counting allocator).
  **DONE — PASS** ([Bench 974](.benchmarks/974_issue974_walk_rank_goat.md)): G2 defaults ≈ 0.28 ms/call on the synthetic 100k view (36× headroom; higher-degree fixture than ndb's — an upper bound); G1 re-run identity ASSERTED on every timed iteration (a determinism regression fails the bench); G4 via the counting allocator in-module (≤ 16 allocs vs 256 steps, the Issue-741 gate posture).
- [x] T4. Record in `.benchmarks/` + this file; demote nothing (both sides ship the same flag name; the ndb re-export is the only consumer until a second appears).
  **DONE** — [Bench 974](.benchmarks/974_issue974_walk_rank_goat.md); the README feature-count sites bumped 720 → 721 (the `count_features` gate green); `cargo refine` miss recorded: `manual_is_multiple_of` (clippy 1.99) is outside the healer's corpus — fixed manually in the ndb bench, file a healer-corpus issue when mining resumes.

## Non-goals

- No RetrievalMode/wiring here — mode vocabulary and store adapters are consumer-side (ndb owns `RetrievalMode::WalkProximity`, `GraphWalkView`, `KgTripleWalkView`).
- No ensemble phase (ndb 640 T6 deferral carries over).

## References

- ndb issue 640 (T1 PoC bench 516 + T4/T5 bench 517) + `riir-neuron-db/.research` lineage: katgpt-rs `.research/640_RT_J_Relational_Context_Recipe.md` (arXiv:2610.17867).
- Precedent: the SST cross-tail substrate lift (katgpt-rs 958 → ndb/ai consumers) — same direction, same flag-name discipline.
