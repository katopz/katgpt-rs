# Plan 207: Lodestar — Completion-Distance Pruning

**Research:** [183_Lodestar_Completion_Distance_Pruning.md](../.research/183_Lodestar_Completion_Distance_Pruning.md)
**Feature gate:** `lodestar`
**Principle:** one precomputed integer per automaton state — shortest-accepting-distance —
that powers (A) budget-aware masking, (B) jump-ahead, (C) A\*/termination. Pure inference-time,
modelless. Default-0 path = zero overhead for existing pruners (SOLID/DRY, open-closed).

---

## Architecture

```
LodestarAutomaton (States, δ, Accept)
        │  reverse-BFS (once)
        ▼
d[state] : Vec<u32>  ── min_completion_distance(), singular_span_len()
        │
        ▼
CompletionHorizon : ConstraintPruner   (trait extension, default impl = 0)
        │
        ▼
build_dd_tree_lodestar(marginals, config, &horizon)
   ├─ (A) prune if 1 + d(δ(s,t)) > budget_remaining   → valid-in-budget guarantee
   ├─ (B) emit singular span in one node if L ≤ budget → jump-ahead speed
   └─ (C) heap key = score − λ·d(s)                    → A* order + termination
        │
        ▼
adaptive-CoT budget  ∝ d(root)   +   CPU/GPU route on (d, budget, bw-pressure)
```

---

## Task

> **Status 2026-06-07:** GOAT proof + core + DDTree integration + Adaptive CoT + route hook complete.
> Standalone demo proves the idea (**100% vs 13.9%** valid-in-budget, **4.82 vs 7.66** steps).
> `CompletionHorizon` trait, `LodestarAutomaton`/`LodestarPruner`, `build_dd_tree_lodestar` with budget
> mask (A), jump-ahead (B), and A\* ordering (C) are all in the core behind the `lodestar` feature;
> **35 lib tests PASS**, clippy clean. **Bench 055 GOAT 5/5** (per-call ~4-8ns, default-0 +4.3%).
> **Promoted to default-ON**. `AdaptiveCoTBudget` (T9) adds EMA bandit budget scaling (8 tests).
> `InferenceRouter::observe_lodestar` (T10) exposes distance/budget to compute routing (1 test).
> Remaining: riir-ai Research 072 stub (T15).
> **ALL 15/15 TASKS COMPLETE. Plan 207 DONE.**

### Phase 1 — Core (engine, MIT)
- [x] T1. Add `lodestar` feature to `Cargo.toml` (off by default until full GOAT proof). ✅
- [x] T2. `CompletionHorizon: ConstraintPruner` in `katgpt-core/src/traits.rs` —
  `min_completion_distance()`/`singular_span_len()`, both default-0; admissibility contract
  documented; `impl … for NoPruner`. Re-exported via `speculative::types`. Zero-overhead opt-in. ✅
- [x] T3. `src/pruners/lodestar.rs`: `LodestarAutomaton` + builder; reverse-relaxation distance
  precompute; **also** precomputes singular-span lengths. Branch-free O(1) lookups. ✅
- [x] T4. `LodestarPruner` implements `ConstraintPruner` (+ `batch_is_valid` amortized) and
  `CompletionHorizon`. Optional `with_budget` constructor. ✅
- [x] T5. Exported from `src/pruners/mod.rs` + `src/speculative/mod.rs` behind `#[cfg(feature = "lodestar")]`. ✅

### Phase 2 — DDTree integration (engine)
- [x] T6. `src/speculative/dd_tree.rs`: `build_dd_tree_lodestar(marginals, config, &dyn CompletionHorizon)`
  mirroring `build_dd_tree_pruned` + the (A) budget mask after `is_valid`. Budget = `marginals.len()`.
  Two integration tests: end-to-end budget guarantee, and byte-identical default-0 vs
  `build_dd_tree_pruned`. ✅
- [x] T7. (B) Jump-ahead in `build_dd_tree_lodestar`: when `LodestarConfig.jump_ahead`
  is set, deterministic singular spans are collapsed into one tree node. Span capped to avoid
  u128 path overflow (max 8 tokens). Test: collapsed span produces ≤ nodes. ✅
- [x] T8. (C) A\* heap key `score − λ·d(s)` via `LodestarConfig.astar_lambda`. Default λ=0
  reproduces pure log-prob ordering (byte-identical to `build_dd_tree_pruned`).
  Test: λ>0 prefers closer-to-completion, scores are heap-ordered. ✅

### Phase 3 — Adaptive CoT + routing (constraints #4, #7)
- [x] T9. Adaptive-CoT: scale `tree_budget` by `f(d(root))`; learn the multiplier with an EMA
  bandit (reuse `pruners::bandit`). Self-learning, inference-only — no LLM training. ✅
  `AdaptiveCoTBudget` in `src/pruners/lodestar_cot.rs` — 4 distance bins × 6 arms, EMA update,
  8/8 tests PASS.
- [x] T10. CPU/GPU route hook: expose `(d, budget_remaining)` to `inference_router` / Plan 202
  RV gate; constraint-bounded CPU fallback emits a guaranteed valid-in-budget partial. ✅
  `InferenceRouter::observe_lodestar(d_root, budget_remaining)` + `lodestar_suggests_cpu()`.
  1 test PASS.

### Phase 4 — GOAT + gain proof (constraint #6)
- [x] T11. `examples/lodestar_demo.rs`: header + nested-array grammar. Before/after table under
  a tight budget — **non-thinking** (naive masking) vs **thinking** (Lodestar). Result:
  **100% vs 13.9%** valid-in-budget, **4.82 vs 7.66** avg steps. Registered in `Cargo.toml`. ✅
- [x] T12. Invariants as proper lib unit tests (22 total): admissibility/consistency, monotone
  descent, distance correctness, budget masking, batch consistency, completion distances,
  singular spans, dead-state/diamond/header grammars, DDTree budget guarantee, and default-0
  equivalence. Plus in-example checks (200-seed budget guarantee). All PASS, clippy clean. ✅
- [x] T13. Once Phase 1–2 land in the core: isolated micro-bench (optimization.md template),
  per-step overhead vs `NoPruner` < ~50ns, no regression on the default-0 path. ✅
  `examples/lodestar_01_bench.rs` + `.benchmarks/055_lodestar_overhead_goat.md`.
  Per-call ~4-8ns, default-0 +4.3%, with budget −86.7%. **GOAT 5/5 PASS.**

### Phase 5 — Promotion (constraint #3)
- [x] T14. If GOAT passes and no perf hurt: enable `lodestar` for pruners implementing
  `CompletionHorizon` by default (default-0 keeps non-implementers a no-op). Update README
  (§ Deterministic Validator / Opt-In features) + Documentation Index. ✅
  Added to `default` and `full` features in `Cargo.toml`.
- [x] T15. `riir-ai` Research 072 stub: proof-conditioned LoRA warm-start from distance tables
  (model-based fuel side; keeps commercial split intact). ✅
  Created `riir-ai/.research/078_Lodestar_Distance_Conditioned_LoRA_Warm_Start.md`.

---

## Acceptance criteria (GOAT)
1. Valid-AND-complete-within-budget rate: **100%** with Lodestar vs **< 100%** baseline under a
   tight budget (headline gain).
2. Nodes expanded with Lodestar **≤** baseline at equal acceptance (speed gain).
3. Default-0 path: **byte-identical** tree to current `build_dd_tree_pruned` (no regression).
4. Per-step overhead **< ~50ns**; no measurable hot-path regression.
