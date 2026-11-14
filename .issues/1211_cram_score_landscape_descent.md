# Issue 1211: Descent-search policy extension for katgpt-assign — K chains, abstain-on-budget, fitted-level stop

> **Status:** OPEN — filed 2026-11-14; no lease (claim per AGENTS.md §claim-lease before implementing)
> **Research:** [katgpt-rs/.research/670_CRAM_Pseudolikelihood_Energy_Descent.md](../.research/670_CRAM_Pseudolikelihood_Energy_Descent.md)
> **Source:** arXiv:2611.17309 "CRAM" (Hoover, Blayney, Whittington, 2026-11-12)
> **Substrate:** `crates/katgpt-assign` (Research 607 / Plan 620) — the driver SHIPS; this issue extends it, never forks it
> **Consumers:** riir-refine Issue 237 (composition cell); riir-grams Issue 001 (Repair act seam)

## What

An EXTENSION of `katgpt-assign`, not a second driver. Substrate-first at filing initially MISSED the shipped local-search driver under the wrong vocabulary ("descent"/"hill_climb" vs its "local search"/"assignment" spellings) — caught at review, re-grepped, now the anchor. What `katgpt-assign` already carries (lib doc, verified): strict-improvement local search (single + swap moves, first-improvement, hot-container ordering, deterministic seeded tie-breaks, move/eval/sweep limits), `AvoidMoving` hard pins (the clamp analog), integer-exact delta evaluation, plateau/escape stages, and simulated annealing as a recorded in-source NEGATIVE. A CRAM cell-to-value problem IS an object-to-container assignment.

The deltas `katgpt-assign` does NOT carry, and this issue adds behind a feature flag:

- **K independent chains + lowest-score pick** — assign solves once; CRAM runs several chains from different random starts and keeps the lowest-cost endpoint (its multi-solution axis; N-Queens/graph-coloring coverage).
- **Abstain-on-budget** — assign returns the best-so-far assignment; a new opt-in return mode gives the caller "no answer" on budget exhaustion instead of a confident wrong (CRAM measured 0% wrong-halts; 51 budget-exhaustions in 1,238,358 searches). Default behavior byte-identical.
- **Fitted-LEVEL stop hook** — the caller supplies a fitted threshold (quantile of cost over solved cases) + stop predicate; calibration stays caller-side (the stack's quantile/conformal doctrine). Level-vs-dynamics is the point: CRAM's level stop never halted at a wrong board, while the five looped baselines' dynamics-based halting rules wrong-halt 0.05–3.54% of stopped puzzles, and EBR's gradient-norm stop fails on Maze entirely.
- **Per-value-pair cost rows (C×C)** — only if `AssignmentAffinities` cannot express the needed cost shape; signal-diff at implementation time, and the spec vocabulary extends only if genuinely unrepresentable (`Problem` never changes shape).

House law: hard moves only (no softmax — sigmoid law); the in-source simulated-annealing negative stands as the prior against any noisy-chain arm (if ever tried, it is an explicitly-gated arm against the hard-move default, expected to lose).

## Proof path (pre-registered; GOAT gate before any promotion; demote the loser)

- Fixture: CRAM-style constraint problems expressed as assign Problems (planted optimum + adversarial local minima + a planted saddle); the chains+stop extension vs assign's incumbent single search at EQUAL total move budget.
- Falsification arm: a deliberately mis-fitted stop threshold MUST yield abstains, never wrong emits — a green run that emits wrong answers FAILS the fixture.
- Gate: chains+stop ≥ incumbent accuracy at equal budget across the fixture family; per-family reporting, never pooled; zero-alloc hot path; assign's existing gates stay green (byte-identical default path — the extension is inert by default).
- If the stop is ever sold as a coverage guarantee → Report-the-Floor (conformal-naive floor, Plan 340) or the gate FAILS.

## Non-goals

No training in katgpt-rs (cost fitting is caller-side / riir-train's job); no game wiring; no ESC changes (riir-rethink's per-suite fitted-level escalation thresholds are the law family — cited, not touched); no cost fitting in the primitive itself.
