# Issue 748 — the Lean proof family has no automatic lane for `develop` work

**Status:** OPEN (documentation-complete, repair owner-gated on Actions spend)

## What was measured (2026-09-12)

A full workstation sweep ran every Lean gate script in the workspace. The
population is **4 repos** — derived with
`git ls-files -- '*.lean' 'lakefile*' 'lean-toolchain'`, never typed, because
the denominator moved from 26 to 31 directories inside one day. Every other
contract repo has ZERO Lean, so "run the proof gates on all repos" is 4 real
gates plus N-4 vacuous.

All six scripts passed, no pin drifted:

| repo | `proof_gate.sh` | theorems vs pin | `proof_negative_test.sh` | `spec_match_gate.sh` |
|---|---|---|---|---|
| katgpt-rs | PASS exit 0 | 39 = 39 | PASS, 8/8 caught | — |
| riir-ai | PASS exit 0 | 16 = 16 | (added 2026-09-12) | — |
| riir-chain | PASS exit 0 | 92 = 92 = 92 | PASS, 9/9 + sentinel | PASS, 12 suites / 107 tests |
| riir-neuron-db | PASS exit 0 | 52 = 52 = 52 | (added 2026-09-12) | — |

Every timing is **warm-cache** (`.lake` already built, 0.67s-34s). No
cold-build number exists for any of the four, and a cold number is not the
compile cost anyway: the two Mathlib repos resolve via `lake exe cache get`,
so "cold" there is a *download*, not a build. Deliberately not measured — a
from-scratch Mathlib build costs hours against a Data volume already at 99%.

## The finding — trigger coverage, not gate correctness

`scripts/ci_gate_coverage.py` asks which declared triggers can actually fire.
Read by hand for this family:

| repo | workflow | triggers | invokes |
|---|---|---|---|
| katgpt-rs | `lean_proofs.yml` | push `[main]`, PR, dispatch | `proof_gate.sh` **only** |
| riir-ai | `lean_proofs.yml` | push `[main]`, PR, dispatch | `proof_gate.sh` |
| riir-chain | `lean_proofs.yml` | push `[main]`, dispatch | `proof_gate.sh` + `proof_negative_test.sh` |
| riir-neuron-db | `lean_proofs.yml` | push `[main]`, dispatch | `proof_gate.sh` |

Two separate gaps, and they must not be pooled:

1. **No lane covers `develop`.** Every trigger is `branches: [main]` under the
   owner call of 2026-09-09 (Actions spending limit). All real work happens on
   `develop`, so for day-to-day work the **workstation run IS the coverage**.
   This is a documented owner decision, not a defect — recorded here so nobody
   reads a green `lean_proofs.yml` badge as covering their develop commit.

2. **katgpt-rs's negative test is invoked by nothing** — not even on `main`.
   riir-chain runs both scripts in one job; katgpt-rs runs only the gate, so
   the one artifact that proves its gate is non-inert never executes in CI.
   This is the asymmetry worth closing.

Note `can fire` is not `does fire`: a `workflow_dispatch` entry is a button,
not coverage.

## Why the obvious repair is NOT applied here

Adding `proof_negative_test.sh` to katgpt-rs's `lean_proofs.yml` is a
three-line change, but it costs **~119s of Actions time per `main` push**
(measured). The owner is actively managing a spending limit and suspended
every schedule in-file for exactly this reason. Spending minutes is an owner
call, so the repair is **gated**, not silently taken.

**Owner decision needed — pick one:**
- (a) add the negative test to katgpt-rs's `lean_proofs.yml` job (~119s/main push), or
- (b) leave CI as-is and rely on the workstation run, or
- (c) make it `workflow_dispatch`-only in that workflow, so it is at least
      nameable without firing on every main push.

## Incidental — a numbering observation, not a claim

`.issues/.highwater` reads **747** and the max issue file ever added in git
history is also **747**, so `numbering_gate.py` reports 0 stale allocators and
passes. But AGENTS.md and HISTORY.md both cite **Issue 750** as a katgpt-rs
issue (the lossy-surface promotion rule, adopted 2026-08-28). Either 750 was
allocated without its file ever being committed, or the allocator regressed
after the file was removed under the noise-reduction rule. The gate cannot see
either case because it only checks `highwater >= max existing file`. Recorded
as an observation; 748 was taken for this issue because it is uncontested
under every source of evidence. (Issue 753 is riir-ai's, not this repo's.)

## Related

- Layer-5 wrap repair, ported to all four this session: katgpt-rs `ad7fc9a7`,
  riir-ai `044445651`, riir-chain `cbe76d3d`. Origin riir-neuron-db Issue 605.
- Negative tests added to riir-ai and riir-neuron-db 2026-09-12, closing the
  "one-sided gate" half of this issue.
