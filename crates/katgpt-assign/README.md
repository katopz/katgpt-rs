# katgpt-assign

[![crates.io](https://img.shields.io/crates/v/katgpt-assign.svg)](https://crates.io/crates/katgpt-assign)
[![Documentation](https://docs.rs/katgpt-assign/badge.svg)](https://docs.rs/katgpt-assign)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Spec-driven **constrained assignment solver** — place every object into
exactly one container such that constraint specs hold and goal specs are
optimized. Pure modelless combinatorics, **zero dependencies**, integer
determinism. Distilled from Meta's Rebalancer (OSDI'24 — Research 607,
Plan 620): a small closed spec vocabulary compiled into an expression DAG
of linear size, searched by strict-improvement local search with delta
evaluation.

## The model

```rust
use katgpt_assign::{Problem, Spec};

let problem = Problem {
    num_objects: 2, num_containers: 2, num_dims: 1,
    demands: vec![1, 1],
    initial: vec![0, 0],
    specs: vec![
        Spec::capacity(0, 1),               // HARD: util <= 1 per container
        Spec::Balance { dim: 0, weight: 1 }, // GOAL: minimize spread
    ],
};
```

Specs (Phase 1 vocabulary):

| Spec | Kind | Fold |
|---|---|---|
| `Capacity { dim, limit, violation_weight }` | hard | per-container row `w·max(0, util−limit)`, all rows folded into one root via **`Max`** (the plan's letter) |
| `Balance { dim, weight }` | goal | `weight · (max_b util − min_b util)` (spread) |
| `MinimizeMovement { weight }` | goal | `weight · #{o : assign(o) ≠ initial(o)}` |

The folded objective the search minimizes:

```
root = NEVER_WORSE_GUARD (10_000) · Max-folded violation
     + Σ goal terms
     + Σ violation rows          (secondary repair pressure; 0 at feasibility)
```

The secondary sum term is identically zero for every feasible assignment
(it never distorts goal semantics) and exists because the Max fold alone
cannot tell WHICH of two tied-at-worst containers to drain — measured,
strict-only acceptance stalls on tie plateaus (the `perfect_balance`
fixture froze at a two-way tie). Every accepted move strictly decreases
it, so it makes plateau descent monotone.

Paper defaults kept: fix-it weight **100**, never-worse guard **10000**.
Simulated annealing is a recorded negative in the Rebalancer source and
deliberately absent.

## Determinism (G1)

No floats anywhere in the objective path, no `HashMap` iteration in the
solve path, dense-array move generation with seeded tie permutations —
**same input + same seed ⇒ byte-identical `Assignment` on every node**.
That is the raw-domain requirement for anything crossing
`SyncBlock → ChainConsensus` (the primary consumer, riir-chain Issue 164,
does exactly that). The one non-deterministic knob is the advisory
wall-clock `Limits::time` (default `None`) — faster machines squeeze more
moves into the same window; replayable results always run time-unbounded.

## Search

- Moves: `Single` + `Swap` (Phase 1; triple-loop and KL-search are
  Phase 2 per Plan 620).
- First-improvement scans, **hot-container ordering** — sources hottest
  first, heat = violation mass + balance deviation (leaf-side
  approximation of the paper's node potentials, the measured decisive
  win, Fig. 6).
- Acceptance window `obj_δ ≤ 0`: strict improvements always; δ=0
  (sideways) moves while a budget lasts, restricted to moves draining a
  container with heat > 0 and never immediately reversing the previous
  sideways move. Sideways moves are how a Max-folded objective traverses
  tie plateaus at all.
- Integer sums are exact — delta evaluation recomputes only reached DAG
  nodes and is bit-identical to a full recompute (pinned by the
  delta-property test; this is why Meta's segment tree is unnecessary
  here).
- Limits: accepted moves / evaluations / sweeps (deterministic) +
  advisory time.

## Layout

- `types` — `Problem`, `Spec`, `Limits`, `Seed`, `Assignment`, `Solution`
- `expr` — the expression DAG (`Lookup`/`Const`/`Sum`/`Max`/`Affine`,
  cached values, parent CSR, delta evaluation)
- `specs` — spec → DAG compilation (leaves shared per `(dim, container)`)
- `solver` — local search (scans, heat, membership indexes, limits)
- `fixtures` — deterministic instance generators + the FFD baseline
- `rng` — SplitMix64
- `alloc` — G4 allocation tracking (`alloc_tracking` feature / debug)

## Posture

`katgpt-core` re-exports this crate as `katgpt_core::assign` behind the
**opt-in** `assignment` feature (the `katgpt-dec` → `katgpt_core::dec`
precedent). Promotion to default requires the GOAT gate (G1+G2 vs
outside baselines) AND a wired consumer (riir-chain Issue 164 minimum) —
Plan 620's verdict protocol.

The leaf `descent` lane (issue 1211 — K chains, abstain-on-budget,
fitted-level gate) is **default-on** since 2026-11-15 (delegated-owner
verdict AGREE; GOAT bench 981 + the riir-grams Repair-act consumer):
compile availability, zero runtime cost unless `solve_descent` is
called. Explicit `--no-default-features` restores the byte-identical
incumbent build.
