# Issue 761 — the restatement-theorem class has a report and no verdict half

**Status:** RESOLVED 2026-09-12 — report AND verdict both landed the same day. T1 unblocked mid-session when a concurrent session closed Issue 750 (`baa7ff8a`), and the resolution changed shape on the way: the verdict half is a **workstation sweep**, not a docs_gate CHECKS row — CI's single checkout would pin the one repo where the class barely exists (katgpt-rs has **1** composite def; riir-neuron-db has 44) and read like coverage. T1 + T2 done, T3 standing by design, T4 deferred on measurement.

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

- [x] **T1 — verdict half.** `scripts/restatement_drift_sweep.py` +
      `scripts/restatement_drift_floors.txt`, 4 rows, all
      `max_restatement = 0` / `max_identity = 0`. **Two floors, and the second
      is the one that bites:** `min_lean_files` catches a WALK regression,
      `min_theorems` a PARSE one — and only the second moves when a tokenizer
      breaks on an unchanged tree. That is not hypothetical: during
      development a `:=` tokenized as `:` + `=` took the def table to 0 with
      the file count identical in both runs. Both floors ~60% of measured.
      ⛔ **Not** a docs_gate CHECKS row (the shape T1 originally specified):
      the sweep family is workstation-only precisely because a single-checkout
      CI run derives a one-repo population and prints a confident green over
      the repos that carry the class.
- [x] **T2 — prove the verdict fires.** Four paths, each exercised on the real
      code rather than argued: `--prove-fires` plants a restatement into a
      COPY of each repo's `.proofs` and requires the count to move (4/4,
      0 → 1); a floor raised above measured REDS (`parsed 58 < floor 99`,
      exit 1); a pinned-but-absent repo REDS as UNSEEN rather than silently
      shrinking the population (exit 1); an empty floors file REFUSES instead
      of passing over zero rows — the absent-vs-malformed trap. Tree restored
      byte-identical after every probe. The historical real-subject probe
      stands too: at `24957a2^` the classifier reports exactly the four
      theorems Issue 617 removed, 0 at HEAD.
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

## What moved while this issue was open

T1's blocker (Issue 750, the CHECKS-array sync) closed mid-session in a
concurrent one. Re-reading the blocker instead of inheriting it is what changed
the deliverable: 750 closing made a CHECKS row *safe*, and thinking about
whether it was *right* is what surfaced that it was neither — CI cannot see the
sibling repos at all. A blocker lifting is a prompt to re-derive the design,
not a green light for the design that was blocked.

## Provenance

riir-neuron-db Issue 617 (`24957a2`, `387b4fc`), which resolved the four
instances; this issue owns the class-level instrument and the sweep. Same
family as Issues 749/751/752/754: a measurement that cannot fail reports a pass
indistinguishable from a real one.
