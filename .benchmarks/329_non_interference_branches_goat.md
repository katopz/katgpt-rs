# Plan 329 — Non-Interference Memory Branches GOAT Gate Results

**Date:** 2026-06-26
**Feature:** `non_interference_branches`
**Plan:** [`.plans/329_non_interference_memory_branches.md`](../.plans/329_non_interference_memory_branches.md)
**Research:** [`.research/310_RIZZ_Non_Interference_Memory_Branches.md`](../.research/310_RIZZ_Non_Interference_Memory_Branches.md)
**Bench:** `crates/katgpt-core/benches/bench_329_non_interference_branches_goat.rs`
**Verdict:** **ALL GATES PASS → PROMOTED to DEFAULT-ON**

---

## Summary

Five generic open primitives shipped behind `non_interference_branches`,
distilled from RIZZ (Goel et al., Oxford, Jun 2026, arXiv:2606.20638). All
GOAT gates (G1–G5) pass with a **pure modelless gain** — no training, no
gradient descent, no learned parameters. The primitives are deterministic
data structures + dot-product projections + budget arithmetic.

| Primitive | Module | Role | Gate coverage |
|-----------|--------|------|---------------|
| `BranchBank<E>` | `branching/bank.rs` | Bounded persistent CognitiveBranch bank (spawn/merge/prune) | G1 |
| `BranchRouter<E>` | `branching/router.rs` | Dot-product snap + Jaccard fallback routing | G1, G2, G4 |
| `VerifierGate` | `branching/verifier.rs` | Reward + curiosity + centroid-quarantine write gate | G1, G4 |
| `NonInterferenceProjection<D>` | `branching/projection.rs` | Orthogonal latent subspace per branch | G1 |
| `BudgetCompiler` | `branching/compiler.rs` | Priority-cascade context compiler under byte budget | G1 |

---

## G1 — Correctness (orthogonality + non-interference + frame-theory limit)

**Status: PASS** — 101 unit tests + 3 bench G1 sub-gates.

### G1a — Pairwise orthogonality (8 canonical basis directions in D=8)

Assigned canonical basis vectors `e_0..e_7` to 8 branches. Verified
`interference(b_i, b_j) < 1e-6` for all 8×7=56 ordered pairs `i≠j`.

```
[PASS] G1a: 8 canonical basis directions in D=8: max pairwise interference =
       0.00e0 (pair b_0↔b_0) < ε=1e-6; is_non_interfering_with_all holds for
       all 8; max_orthogonal_branches=8
```

The max pairwise interference is exactly 0 (canonical basis is perfectly
orthogonal by construction). `is_non_interfering_with_all(b_i)` holds for
every branch. `max_orthogonal_branches() == D == 8` (frame-theory limit).

### G1b — Non-interference by construction (write to b_i, b_j unchanged)

Spawned 8 branches with canonical-basis anchors. Wrote one episodic entry to
branch 0. Verified branch 0's episodic store grew 0→1, and **every other
branch's episodic/procedural/failure stores are byte-for-byte unchanged**.

```
[PASS] G1b: write_episodic(b_0) grew b_0.episodic 0→1; branches b_1..b_7
       stores unchanged (non-interference by construction)
```

This is the RIZZ headline: writes to one branch do not contaminate sibling
branches. The guarantee is structural (separate Vec stores per branch), not
learned.

### G1c — Frame-theory limit (9th direction in D=8 must interfere)

Assigned all 8 canonical basis directions, then attempted to assign a 9th
direction = normalized all-ones vector `(1,1,...,1)/sqrt(8)`. The dot-product
of this with every `e_k` is `1/sqrt(8) ≈ 0.354 > 0.1` threshold.

```
[PASS] G1c: 9th direction (uniform) in D=8 correctly rejected: interferes
       with b_0 by 0.3536 ≥ 1/sqrt(8) = 0.3536 > threshold 0.1
```

`assign_direction` correctly returned `AssignError::Interferes` with the
worst-offender branch id and the measured interference magnitude. This proves
the orthogonal capacity limit is enforced: beyond `D` branches, any new
direction must interfere by ≥ `1/sqrt(D)`.

---

## G2 — Perf (router hot path)

**Status: PASS**

Measured via `bench_329_non_interference_branches_goat` (release, 10,000
iters after 1,000 warmup, `std::time::Instant`, `black_box`).

| Primitive | Target | Steady-state | Margin |
|-----------|--------|--------------|--------|
| `BranchRouter::route` (64-branch bank, D=8) | ≤ 1,000ns (1µs) | ~301.5ns | 3.3× under |

```
[PASS] G2: BranchRouter::route median ~301.5ns (≤ 1000ns target) over 10000
       iters on 64-branch bank (D=8); resolved to b_5
```

The router walks all 64 active branches via a branch-free max-reduction dot-
product scan (no early exit). 301.5ns is well within the per-tick per-NPC
budget. The `VerifierGate::should_write` is three scalar comparisons with
early returns — sub-nanosecond, not separately gated.

---

## G3 — No-regression (feature-combination build)

**Status: PASS**

```
cargo check -p katgpt-core --all-features           # clean
cargo check -p katgpt-core                           # clean (default, post-promotion)
cargo check -p katgpt-core --no-default-features     # clean
cargo check                                          # clean (root crate, post-promotion)
```

The `--all-features` check catches the merkle_root class of combo-only bugs
(no single feature turns on both `non_interference_branches` and every other
feature simultaneously). The `--no-default-features` check confirms zero
overhead when the feature is off (every public API is `#[cfg]`-gated).

---

## G4 — Alloc-free hot path

**Status: PASS**

Measured via `CountingAllocator` over 100 steady-state calls (1 warmup call
excluded).

| Primitive | Allocs / 100 calls | Status |
|-----------|-------------------:|--------|
| `BranchRouter::route` | 0 | PASS |
| `VerifierGate::should_write` | 0 | PASS |

```
[PASS] G4: BranchRouter::route: 0 allocs / 100 calls; VerifierGate::should_write:
       0 allocs / 100 calls
```

The router's dot-product scan reuses the bank's pre-allocated slot array via
`active_branches()` iterator (filter on lifecycle, no Vec allocation). The
verifier is three scalar comparisons. A companion gate (G4b) verified the
0-alloc result is non-degenerate by confirming the gates return the correct
`WriteDecision` variants for known inputs:

```
[PASS] G4b: VerifierGate returns Write/Quarantine/Reject (reward-low) /
       Reject (curiosity-low) for known inputs — 0-alloc result is non-degenerate
```

The cold-path lifecycle operations (`spawn`, `merge`, `prune`,
`assign_direction` on a fresh row, `BudgetCompiler::compile`) DO allocate —
by design. They run on branch lifecycle transitions, not per-tick.

---

## G5 — Modelless

**Status: PASS** — verified structurally.

The `non_interference_branches` feature has `[]` dependencies in
`crates/katgpt-core/Cargo.toml`:

```toml
non_interference_branches = []
```

No `riir_train`, no `riir_gpu`, no training-loop dependency. The five
primitives are pure:
- `BranchBank`: pre-allocated Vec + free-list (data structure).
- `BranchRouter`: dot-product max-reduction (closed-form arithmetic).
- `VerifierGate`: three scalar comparisons (closed-form).
- `NonInterferenceProjection`: dot-product projection + L2 normalization
  (closed-form linear algebra).
- `BudgetCompiler`: priority cascade with byte accounting (closed-form
  arithmetic).

No weight mutations, no backprop, no gradient descent. The only "state
update" is appending to pre-allocated Vecs (episodic/procedural/failure
stores) and writing normalized direction vectors into pre-allocated matrix
rows — both are deterministic data-structure operations, not learning.

---

## Promotion Decision

**PROMOTED to DEFAULT-ON** (Phase 3, 2026-06-26).

All GOAT gates pass (G1–G5) with a pure modelless gain:
- No training, no gradient descent, no learned parameters.
- Five deterministic data structures + closed-form arithmetic primitives
  distilled from RIZZ.
- Zero-alloc per-tick hot path (route + should_write); lifecycle operations
  are alloc-tolerant by design (cold path).
- The gain is a new capability class — per-NPC continual adaptation without
  catastrophic interference — not a perf optimization on a biased answer.
- The non-interference guarantee is structural (geometric orthogonality),
  not learned. This is the strongest possible GOAT verdict: the correctness
  property holds by construction, not by training convergence.

The feature is now on for every consumer of `katgpt-core` by default.
Downstream impact is pure additive — the module compiles but does nothing
unless a caller invokes the primitives.

### Post-promotion verification

```
cargo check -p katgpt-core                     # clean (default now includes the feature)
cargo check                                    # clean (root crate default)
cargo test -p katgpt-core --lib branching      # 101/101 pass (no explicit feature flag needed)
```

---

## Composition note

When `arg_protocol` is also enabled (now also default-on, Plan 327),
`BranchLifecycle` becomes a type alias for `crate::arg::LifecycleState` —
the same enum used by the ARG protocol's ontology lifecycle (Step E). This
makes branch lifecycle state committable and redirect-resolvable via the ARG
`RedirectTable`. The composition is verified clean:

```
cargo check -p katgpt-core --features non_interference_branches,arg_protocol  # clean
cargo test -p katgpt-core --features non_interference_branches,arg_protocol --lib branching  # 101/101
```

Phase 4 composition tests (BranchBank × ARG, VerifierGate × CLR,
BranchRouter × Engram, NonInterferenceProjection × closure-instrument) are
the next work item.
