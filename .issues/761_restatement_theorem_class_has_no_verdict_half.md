# Issue 761 — the restatement-theorem class has a report and no verdict half

**Status:** OPEN — the **report** landed 2026-09-12 (`scripts/restatement_theorem_audit.py`, first measurement 0 findings over 4 repos / 68 `.lean` / 255 theorems, oracle-validated two-sided against riir-neuron-db `24957a2^`). What is missing is the half that makes a zero *stay* zero: a floors file and a per-push check. Deliberately deferred, reason below.

## The class

A Lean spec theorem whose RHS is its own `def` body:

```lean
def sidecarHeaderSize : Nat :=
  magicSize + versionSize + envelopeRefSize + marginSize + variantSize + nFactsSize

theorem sidecarHeaderSize_eq_sum : sidecarHeaderSize =
    magicSize + versionSize + envelopeRefSize + marginSize + variantSize + nFactsSize := by decide
```

`decide` discharges it for **any** constant values. Four shipped in
riir-neuron-db for months and were removed by its Issue 617 (`24957a2`) — with
doc comments claiming each was "the `merkle_root` guard", i.e. the exact bug it
provably could not see. Nothing in the stack could tell it apart from a theorem
that proves something: `lake build` green, `#print axioms` axiom-free,
`proof_gate.sh` counting it in the audited surface, `proof_negative_test.sh`
17/17 — because no arm ever red on it, and a pass count cannot report which
theorem *didn't* fire.

## What landed (the report half)

`scripts/restatement_theorem_audit.py` — a report, exit 0, population derived
(BOUNDARY.md + `.git` + `.proofs`). Criterion: symbolic equality over **leaf**
constants. Buckets and the reasoning for each boundary are in its docstring and
the AGENTS.md section; the two that matter:

- **CROSS-DEF is not a finding.** `commitmentOffset = RAW_PREFIX_LEN` is
  symbolically equal too and is *load-bearing* (a perturbation arm reds on it):
  it pins two independently maintained definitions. Pooling would have
  condemned it.
- **UNRESOLVED is not clean**, and **HYPOTHETICAL** (binder-carrying) is split
  out rather than pooled — it is 199 of 255, and pooling hides how few
  statements the arithmetic pass ever sees.

Validated against a tree whose answer was known by other means: at
`24957a2^` it reports **exactly** the four removed theorems, 0 at HEAD. That
run is also what exposed the classifier's own soundness bug — a repo-wide def
table unfolded `Shard.zoneHashOffset` through ExperienceGraph's same-named
definition, and only the caveat NAMING the wrong repo's symbol gave it away.

## Tasks

- [ ] **T1 — verdict half.** `scripts/restatement_floors.txt` (per repo:
      `max_restatement`, plus the population floors `min_lean_files`,
      `min_theorems` — a ceiling cannot fail once the walk goes blind) +
      `restatement_floor_gate.py`, and a CHECKS row in `docs_gate.sh`.
      **Blocked on Issue 750**: that array is under concurrent edit, and a
      membership pin written into a moving array re-introduces exactly the
      count drift this family exists to catch. Land after 750 settles.
- [ ] **T2 — prove the gate fires.** A revert probe on the real subject:
      restore one theorem from `24957a2^` into a `/tmp` copy and require the
      gate to RED. A pin that has never failed is a pin of unknown width
      (`a-green-canary-may-be-inert`).
- [ ] **T3 — re-read the UNRESOLVED census when the population moves.** 19
      rows + 6 conjuncts were read one by one on 2026-09-12 and are all value
      pins (function application over a pattern-match def, record projection,
      ℚ division, Mathlib). That census was adjudicated by **reading the
      source**, not by an independent instrument — it bounds reader error
      only, in the direction somebody thought to look.
- [-] **T4 — a `∧`-conjunct is only checked when every conjunct is an `=`.**
      Deferred on measurement, not on difficulty: 2 such theorems exist
      workspace-wide and both decompose to UNRESOLVED conjuncts. Re-open if
      the count grows.

## Provenance

riir-neuron-db Issue 617 (`24957a2`, `387b4fc`), which resolved the four
instances; this issue owns the class-level instrument and the sweep. Same
family as Issues 749/751/752/754: a measurement that cannot fail reports a pass
indistinguishable from a real one.
