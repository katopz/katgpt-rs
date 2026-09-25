# Issue 892 — tetris strategy RULEBOOK (ruliology surface) + moka-PUCT chance-node search + laya head-to-head

**Status: OPEN — T0 landed (shared lookahead module extracted, byte-identical output); T1–T5 in flight.**

Owner directive 2026-09-25 (after Bench 891): encode the owner's Tetris
technique list as a **ruliology surface** — rules as enumerable DATA (KG
triple + precondition + feature + weight), an FSM over play modes, and a
mutable genome — "so we can use it later like we did before for self
evolve"; transplant the moka+PUCT trick; then run the laya head-to-head
before and after.

## Precedents consumed (substrate-first)

- **Ruliology ↔ self-evolve**: riir-clippy `src/self_evolve.rs` +
  `src/ruliology_search.rs` (rules as data, Pareto selection copied from
  `katgpt-ruliology::WinMatrix::pareto_front`) + `src/kat_promotion.rs`
  (Promote / Reject / Withdrawn lifecycle against a frozen score-bench).
  riir-ai Plan 575 (attack combo DAG as a YAML rulebook + Elo book) and
  Plan 576 ("FSM thought ruliology, not per-scene rules").
- **katgpt-ruliology** (`crates/katgpt-ruliology`): `SimpleProgram::id()`
  is BLAKE3 over the behaviour; enumeration + tournament + Pareto; mutation
  `OutputFlip` / `EdgeReroute`; `delta_gated_co_evolve`. The FSMs are
  binary-action, so the Tetris rulebook does NOT reuse `FsmStrategy`; it
  reuses the SHAPE (enumerate subsets → tournament → Pareto on
  (score, complexity) → blake3 genome id).
- **moka+PUCT trick** (Bench 205: 74% → 98% vs greedy Moka): policy PRIOR
  pruned to `top_k` + VALUE at the leaf, no rollouts, most-visited root.
  Negatives: UCB1 without a prior ≈ 0%; opening-book overrides (Bench 204)
  and engram fusion (Bench 848, positions never repeat) failed. The only
  PUCT is Go/Moka-bound (`katgpt-moka-wasm/src/puct.rs`,
  `katgpt-pruners/src/go/moka_net.rs`); `katgpt-core::mcts` is UCB1 +
  rollouts (the failing variant); NO chance-node / expectimax primitive
  exists anywhere.
- Plan 607 deferred PUCT for Tetris on the premise that the gap to laya is
  rendering, not search. Bench 891 refuted the premise in the survival
  regime (depth-2 ≈ 2× survival under garbage) — this issue re-opens it.

## Tasks

- [x] T0 — extract `examples/common/tetris_lookahead.rs` (eval, pickers,
  seeded 7-bag with `remaining()` = exact chance support, garbage board);
  `tetris_05` consumes it — output byte-identical (6 games × 400 × 16 rows).
  Garbage fill is a CLI arg now: Bench 891's prose says 85%, the committed
  code ran 75% — every future row states its fill.
- [ ] T1 — `examples/common/tetris_rulebook.rs`: the owner's techniques as
  a rule table (id, KG triple, source clause, physics precondition,
  per-mode weight, feature), a play-mode FSM (Build / Downstack / Survive)
  with data transition predicates, and a `Genome` (enable mask + weights,
  blake3 id, one-line text round-trip) — the self-evolve surface.
- [ ] T2 — `examples/tetris_06_rulebook_arena.rs`: ruliology enumeration
  over the toggleable rule subsets (2^k) on seeded garbage games, ranked +
  Pareto front on (survival, lines, complexity = enabled rules); plus a
  delta-gated hill-climb over weights (accept only strict improvement ≥ δ).
- [ ] T3 — generic chance-node PUCT primitive (the moka trick for
  single-player stochastic games): prior = sigmoid-normalised scores
  (sigmoid, not softmax), `top_k`, value leaf, chance nodes over an
  explicit outcome distribution (the 7-bag remainder), no sign flip.
  Feature-gated opt-in; GOAT gate (G1 Q-sign/chance-weighting tests,
  G2 latency, G3 no-regression).
- [ ] T4 — rulebook player variants in the arena: depth-2 vs depth-3
  (bag-aware) vs PUCT, hold on/off (hold is NOT legal in the laya arena —
  measured separately, never in the head-to-head).
- [ ] T5 — laya head-to-head on identical seeds (`tetris_07_laya_h2h`):
  baseline (Bench 891 ply2-shaped) BEFORE, rulebook champion AFTER.
- [ ] T6 — benchmark record + promote/demote per GOAT verdict.

## Rule applicability (physics is part of the rule)

The arena physics is `DropRule::FromTop` (real hard drop, no soft-drop, no
rotation during descent). Under it:
- **Rotate both directions** is NATIVE — the placement enumerator already
  reaches every rotation in one step; it matters only for a real-time
  input controller (finesse), recorded as a rule with no board weight.
- **T-spins** are UNREACHABLE — a T-spin needs a rotation into a covered
  slot during descent; a hard drop from the top never enters a covered
  slot. Recorded with precondition `physics = SoftDropRotate`, inactive
  under `FromTop`. Not deferred silently: the precondition IS the record.
- **Hold** is a legal rule in our sim, absent from the laya arena.
