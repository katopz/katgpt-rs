# Bench 698 — TPR binding algebra GOAT gate (`tpr`, Issue 707 / Research 527)

**Status:** ALL PASS (G1, G2, G3, G4, G8) — measured 2026-09-02 on M3 Max
(macOS 26.6.2, release profile). Feature stays **opt-in**: the no-default-consumer
rule blocks promotion until a real consumer lands, not the gates.

**Instrument:** `crates/katgpt-core/benches/bench_707_tpr_binding_goat.rs`
(`cargo bench -p katgpt-core --no-default-features --features tpr --bench bench_707_tpr_binding_goat`)
**Primitive:** `katgpt_core::tpr` (Issue 707 — closed + removed 2026-09-03, its record is the last section of this file; `.research/527`, arXiv:2608.29530)

## Verdict

| gate | claim | measured | verdict |
|---|---|---|---|
| G1 | planted TPR recovered; double fit bit-identical; returned fit = trajectory min; holdout unbind + surgery | energy fraction **2.28e-9** (bar 1e-6); byte-identical artifacts; `fit_objective == min(ssr_per_sweep)`; cos min **0.999999**; surgery max |Δ| 9.5e-7 at scale 3.95 | **PASS** |
| G2 | surgery p99 < 1 µs at D ∈ {64, 256, 768}; projection ≤ 2× its two-GEMV floor; ALS ≤ GD wall-clock | 84 / 167 / **417 ns** p99; **1.31–1.34×** floor; ALS **11.2 ms** vs GD **28.5 s** | **PASS** |
| G3 | opt-in + `RIIR_TPR=0` disables every op | `tpr` absent from the default list; child process with the var set refuses bind/unbind/surgery/project/encode | **PASS** |
| G4 | zero steady-state allocations | **0** across 20 000 op quadruples | **PASS** |
| G8 | withheld-pair top-1 beats the atomic-dictionary null by ≥ 20 pp **on an interpretable corpus** | TPR **100.0%** vs null **0.0%** (chance 1.8%, pool 56) — null ID coverage 100%, ID top-1 100%; composition roles/filler max **3** mean **2.000** over 8 fillers, pool coverage **100.0%** | **PASS** |

Three consecutive runs: project ratio 1.33 / 1.34 / 1.31×, surgery p50 identical
(42 / 125 / 333 ns). Nothing here is a single-run number.

## What the perf work actually was (three layouts, all measured)

The first working implementation FAILED G2 on surgery: **p50 2708 ns at D=768**
against a 1 µs bar. Three layout/kernel changes, each driven by the measurement
rather than by taste:

| encoder layout / kernel | bind p50 @ D=768 | why |
|---|---:|---|
| `D × (m·d)` row-major, hand-rolled dot | 2708 ns | strides one `d`-run per row: touches `m×` the cache lines it reads |
| block-major `D × d`, hand-rolled dot | 2625 ns | contiguous, but each row is a length-`d` REDUCTION — and a 4-way unroll with ONE accumulator does not break the dependency chain |
| block-major `D × d`, `simd_dot_f32` | 1042 ns | the house NEON kernel does break it |
| **block-major, `d × D` column-slice + axpy** | **333 ns** | `d` contiguous axpys, no reduction at all, `out` stays in L1 |

The column-slice form is what ships (`TprArtifact::w`), and it costs the cold
ops nothing: decode is `K` axpys and `Wᵀ(e−b)` is `K` contiguous length-`D`
dots, both contiguous.

**The projection needed a separate fix, and the first attempt made it worse.**
After the layout work, projection sat at 2.14× its floor. A hand-rolled
4-accumulator column dot — the obvious next step — measured **3586 ns, worse
than the 1947 ns it replaced**: the scalar loop did not vectorize where the
house kernel did. The actual cost was elsewhere: `chol_solve_f32` at K = 32 is a
**32-step serial triangular solve**, latency-bound against a bandwidth-bound
GEMV floor. Shipping the explicit `(WᵀW + λI)⁻¹` (via `linalg::spd_inverse_f32`,
which still factors the Gram, so positive-definiteness is still checked at fit
time) turns the readout into one K×K matvec: **1947 → 1222 ns, 1.33× floor**.

Read that gate carefully: its verdict is decided by how fast the FLOOR arm is,
not only by the projection. The floor is two GEMVs executed in the *fastest*
available direction, so 1.33× is a claim about the projection against the best
possible execution of the same FLOP count — not against a strawman.

## The ALS-vs-GD arm is a real comparison, and it is lopsided

The bench carries a full-batch gradient-descent fit of the SAME objective as its
loser arm (bench-local; nothing gradient-based ships). ALS reaches
`ssr 1.39e-4` in **11.2 ms**. GD burns its 20 000-iteration budget in **28.5 s**
— 2500× the wall clock — and **never reaches the ALS objective** (stalls at
3.38e-4). The gate is "ALS ≤ GD wall-clock"; the honest statement is stronger:
GD did not get there at all within the budget.

## Two instrument bugs found by disbelieving a number

Both would have shipped a wrong verdict, and neither was in the primitive:

1. **The null looked vacuous when it was not.** G8 first reported the atomic
   dictionary at **17.9% in-distribution** — the signature of a vacuous null
   (riir-clippy `.benchmarks/062`). It was the harness: the memorizer was
   scored against a ~700-candidate pool while TPR faced 56. Same-shape pools →
   the null is at **100% ID**, which is what makes its OOD zero *informative*.
   A vacuous null certifies nothing; a real one does, and the difference was a
   pool-construction detail.
2. **The OOD arm measured the fixture, not the primitive.** An early
   `withheld_pair_top1` read 50%. The candidate pool had been built from ONE
   test state's bindings, so the other states' truths were not in it — those
   states were unanswerable by construction. Per-state reconstruction errors
   showed the model discriminating by 4 orders of magnitude the whole time
   (0.0006 vs 4.36). Fixed by building the pool over every test state; the
   contract is now documented on the function.

## Also measured, not assumed

- **The monotone certificate is enforced, not observed.** Blocks 3–4 minimize a
  core-space surrogate, so an uphill sweep is possible — one was measured
  (2e-9 on a 1.96e-5 objective, i.e. f32 noise at the convergence floor, 7
  orders below where the fit started). The fit now REJECTS such a proposal and
  rolls back, so the artifact is the minimum of the recorded trajectory, and
  the bench asserts exactly that (`fit_objective == min(ssr_per_sweep)`) rather
  than asserting a counter is zero.
- **The kill switch is verified by re-exec.** `RIIR_TPR` is `OnceLock`-cached,
  so an in-process check after any other gate would test the cache. The bench
  re-runs its own binary with the variable set and reads the child's verdict.
- **`types.rs` had never been compiled** before this pass: three `arity.max(1)`
  calls on a `&usize` (E0308). A file can look finished and be unbuildable.

## Scope limits

- Corpora are planted TPRs — G8 is a **positive control**, not evidence about
  any real latent family. The real-corpus arm is riir-clippy Issue 062 T4
  (healer/`rustc_errors` retrieval), whose measured OOD baseline is already in
  `.benchmarks/062_withheld_pair_ood.md` and whose caveats (vacuous healer null,
  step-function in filler count, role/filler collinearity) bound what T-G8 may
  claim there.
- One binding per role block, `n_slots ≤ m` (the square unbind case). Ragged
  and over-complete role sets are unfitted.
- G2 is M3 Max only. The layout wins are cache/dependency-structure wins, so
  they should carry, but that is a prediction, not a measurement.

## G8's corpus is now MEASURED healthy, not asserted healthy (Issue 711 T4)

G8 compares a withheld-`(role, filler)`-pair OOD top-1 against an atomic null.
Two things can make that number unreadable rather than wrong, and until
`4af2b3cf` neither was measured here:

1. **The pool** — a truth absent from the shared candidate pool cannot be
   scored correctly, so `top1` is bounded by `candidate_pool_coverage` and must
   be read against *that*, not against `1.0`. This is the failure that already
   bit this gate once (§"the pool had been built from ONE state's bindings").
2. **The composition covariate** — when every filler is seen with exactly one
   role, withholding a *pair* withholds the whole filler, so the OOD arm is not
   a harder version of the ID arm but a **different question**. `withheld_pair_top1`
   is the probe this hits hardest.

`withheld_pair_top1_report` returns both beside the number, and G8 now gates on
`readable = verdict().is_some() && coverage > 0.99` in addition to the margin.
Measured on the planted corpus: roles/filler **max 3, mean 2.000 over 8
fillers**, pool coverage **100.0%** → G8 PASS with margin 100.0 pp. The prose
claim "G8's corpus is planted and healthy" is now a printed quantity.

**The guard was canaried, not assumed.** Forcing the coverage threshold to
`> 1.5` makes the gate print `G8 corpus is NOT interpretable (readable =
false)` and exit non-zero with `G8 FAIL` / `Issue 707 GOAT gate: FAIL` — so the
new condition is wired to the verdict rather than printed beside it. Restored
immediately; a guard that has never been observed to fire has not been tested.

**Why the raw API did not change.** `withheld_pair_top1` still returns a bare
`f32`. The Issue 711 T4 question was framed as "should the probe REFUSE?", with
refusing described as honest but a breaking change to a gate this GOAT depends
on. The additive shape makes it neither: the number stays available to a
consumer that has already checked its corpus, the refusal is available to one
that has not, and the *gate* — which owns its own corpus — takes the strict
reading. Same resolution T1–T3 used for `bow_router` / `shuffled_role_control`
(`verdict() -> Option<_>` beside the raw bool), so the four instruments now
answer the interpretability question the same way.

## Issue 707 — closed + removed 2026-09-03; this is its record

The PoC issue (`.issues/707`, last revision `b7913d79`; recover with
`git log --all -- '.issues/707_*.md'`) shipped everything the primitive can
prove alone: Phases 1–3 landed in `1f7a96d4`, the harness hardening in
`e079dc7c` / `4af2b3cf` / `8c7ca74b` (Issue 710/711), and all four T-G8 clauses
carry measured evidence — the intervention-battery clause through riir-ai's
real `quest_grammar` consumer (`c4222f983`, Bench 852). Every task row was `[x]`
except the owner-gated deferrals below, which move here so the gate table is
not read as complete.

### Where the shipped primitive diverges from the issue's task text (all measured)

1. **T4 ships the explicit `(WᵀW + λI)⁻¹`, not the Cholesky factor.** A K=32
   triangular solve is a serial chain and put projection at 2.14× its floor
   against a 2× bar; one K×K matvec: 1.33×. Definiteness is still checked at
   fit (`linalg::spd_inverse_f32`).
2. **T5's "HOSVD/QR init" is a seeded filler table + an exact `W,b` solve.**
   The pivoted QR that did ship is the unbind basis (T2), where it is
   load-bearing. `tucker_factorization` is deliberately not a dep.
3. **The monotone certificate is ENFORCED, not counted.** An uphill ALS sweep is
   possible (one was measured); the fit rejects and rolls back, so
   `fit_objective == min(ssr_per_sweep)` is the checkable claim.

### Consumer ledger — three landed, none default-path

| consumer | verdict | what it taught the primitive |
|---|---|---|
| riir-clippy Issue 062 → Bench 063 (2026-09-02) | **FLAT** — `bow_router` 1.14 / `cross_state` 1.07 on the 256-D control vs `r_bow < r_full` in the 8-D space the reranker scores in; re-scoped to a scalar bridge | Issue 710: `shuffled_role_control` could not fail on a single-binding corpus. Fixed `af3bca25` — vacuity is a reported quantity on all four instruments |
| riir-ai Issue 847 Phase 2 → Bench 852 (`c4222f983`, 2026-09-02) | **GOAT 10/10**, opt-in `quest_tpr` — surgery additivity 1.788e-7, validator cascade 12/12 + 8/8 + 4/4 refused + 2/2, withheld-pair composition 8/8 vs 0/8 memorization | the binding-level boundary: withheld-(filler, role) does **not** extrapolate (0/2 at 0.90) — construction generalizes novel combinations of *seen* bindings, not novel bindings |
| riir-clippy Issue 062 T4 → Bench 065 (`cd6c068`, 2026-09-03) | **REFUTED** — ID top-1 21.1 → 46.8 → 63.3 → 81.1% across the `d` grid while OOD stays at chance; margin 0.0 pp vs the 20 pp G8 bar on a readable instrument | corroborates 847's wall from an independent domain and harness. Not a weakening of G8 — G8 is a *planted positive control*; this is the negative real-corpus companion it lacked |

**Two consumers, two domains, two harnesses, same wall.** That bounds what any
future default-path consumer may claim. riir-clippy's Bench 062 measured the
OOD baseline T-G8 had to beat and three constraints on it: use `rustc_errors`
(79.1% ID / 0.0% OOD) as the informative atomic-null arm, never the healer
corpus (0.0% ID — vacuous); scope claims to retrieved-but-demoted rows (OOD
accuracy is a step function in a role's remaining filler count); the corpora
are nearly role/filler-collinear, so widening fillers-per-role is a corpus task.

### Deferred rows — owner decisions taken 2026-09-03

- [x] **T-Promote — DECIDED: NOT promoted; `tpr` stays opt-in.** Gates ALL
  PASS, and that is exactly why this needed a decision rather than a wait:
  three consumers landed (flat, GOAT-but-opt-in, refuted) and two of them
  measured the same generalization wall from independent domains. Flipping
  `tpr` into `default` would compile a rank-m binding algebra into every
  katgpt-core consumer for zero default-path gain — the no-default-consumer
  rule is the SOLID call, not a blocker. **Reopen conditions (any one):** a
  consumer wants `tpr` on ITS default path and its GOAT beats the memorization
  baseline on the binding axis, not just the composition axis; or a simpler op
  wins the slot, in which case demote-loser applies to `tpr` itself.
- [-] **T8** OOD withheld-pair eval protocol + L2,1 arm A/B for trained
  artifacts. **Filed where its unblock lives: riir-train `.issues/505`** — the
  protocol is `withheld_pair_top1_report` + `AtomicNull` + the readability
  helpers (already shipped, and the shape `run_g8()` above runs), NOT
  `validate_bindings`, which holds out **states** and is in-distribution on the
  pair axis; the trigger is the next training run touching a matrix that must
  compose systematically **on a corpus that crosses roles and fillers** — a
  collinear corpus returns chance OOD as a property of the corpus (Bench 065
  §8). The L2,1 half is untested anywhere: `run_g8()` fits at the
  `AlsConfig::new` default `L21::Off`, so the 100% PASS says nothing about the
  regularizer, and neither existing corpus has the headroom to (G8 is at
  ceiling, riir-clippy's at chance). Issue 505 carries the corrected pointer,
  the three-corpus prior and the blocking preconditions.
- [x] **T9 — CLOSED, won't build speculatively.** A TPR surrogate over the
  riir-clippy drafter is an offline interpretability study whose one concrete
  hook (Bench-694 markov-head contract) is answerable by direct inspection.
  Bench 065 already showed the healer corpus is a memorization surface for
  TPR; a surrogate fit there would report the same wall. Reopen only with a
  question the surrogate answers that inspection cannot.
- [x] **T10 — CLOSED, won't build speculatively.** A runtime guide for NPC
  personality surgery presupposes a production consumer; riir-ai 847 is that
  consumer's owner and carries its own Promote row. Writing the guide before
  the consumer exists is documentation of nothing (the Issue 003 shape).
