# Issue 974 — the walk-proximity context-ranker primitive (lift from riir-neuron-db issue 640)

**Status:** Open — filed by the ndb T7 ledger (ndb issue 640 T1–T5 all PASS: mechanism bench 516, G2 latency bench ndb 517). The primitive is PROVEN ndb-side and ships standalone over a 2-method trait; this issue is the lift to `katgpt-core` so every downstream consumes ONE copy (per the per-stack ledger / the ndb issue's constraint: "Do not fork the math ndb-side permanently"). Modelless: BLAKE3 + splitmix64 + xorshift64 + integer counting — no weights, no gradients.

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

- [ ] T1. Move the math (trait + stage-1 + stage-2 + the rng/seed helpers) into `katgpt-core` behind opt-in `walk_rank` (no new deps — blake3 is already in-tree); ndb's `walk_rank` feature re-exports at the same paths (the Plan 319 consumer shape), feature implied by ndb's own flag.
- [ ] T2. Port the 4 in-module tests + the KG/NS adapter test pattern (adapters stay CONSUMER-side — the trait means the lift moves no graph code).
- [ ] T3. GOAT: G1 determinism/leak suite + G2 latency surface re-run katgpt-core-side (bars: byte-identical re-runs; defaults ≤ 10 ms at 100k nodes — ndb 517 measured 0.13 ms worst cell, so the bar has 75× headroom); G4 alloc: once-per-call scratch, zero per-step alloc (assert with the counting allocator).
- [ ] T4. Record in `.benchmarks/` + this file; demote nothing (both sides ship the same flag name; the ndb re-export is the only consumer until a second appears).

## Non-goals

- No RetrievalMode/wiring here — mode vocabulary and store adapters are consumer-side (ndb owns `RetrievalMode::WalkProximity`, `GraphWalkView`, `KgTripleWalkView`).
- No ensemble phase (ndb 640 T6 deferral carries over).

## References

- ndb issue 640 (T1 PoC bench 516 + T4/T5 bench 517) + `riir-neuron-db/.research` lineage: katgpt-rs `.research/640_RT_J_Relational_Context_Recipe.md` (arXiv:2610.17867).
- Precedent: the SST cross-tail substrate lift (katgpt-rs 958 → ndb/ai consumers) — same direction, same flag-name discipline.
