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
| riir-ai | PASS exit 0 | 16 = 16 | PASS, 8/8 — **added** `8b0de0f64` | — |
| riir-chain | PASS exit 0 | 92 = 92 = 92 | PASS, 9/9 + sentinel | PASS, 12 suites / 107 tests |
| riir-neuron-db | PASS exit 0 | 52 = 52 = 52 | PASS, 9/9 — **added** `41f3dcd` | — |

The two new scripts were re-run independently of the session that wrote them
(8/8 and 9/9, exit 0, sentinel on each file's own last line, `.proofs`
byte-clean after). Both carry the harness-bug guard **proven to fire**: a
deliberately vacuous sed reports `HARNESS BUG` and exits 1, so a perturbation
that silently matches nothing cannot pass as a rejection.

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

2. **Three of the four negative tests are invoked by nothing** — not even on
   `main`. Only riir-chain runs both scripts in one job. katgpt-rs, riir-ai and
   riir-neuron-db run only `proof_gate.sh`, so in each the one artifact that
   proves the gate is non-inert never executes in CI.

   This gap WIDENED on 2026-09-12: riir-ai and riir-neuron-db previously had no
   negative test at all, and adding one to each turned "nothing to invoke" into
   "something to invoke that nothing invokes". Writing the script is the
   cheaper half; wiring it is the owner-gated half.

Note `can fire` is not `does fire`: a `workflow_dispatch` entry is a button,
not coverage.

## Why the obvious repair is NOT applied here

Adding `proof_negative_test.sh` to katgpt-rs's `lean_proofs.yml` is a
three-line change, but it costs **~119s of Actions time per `main` push**
(measured). The owner is actively managing a spending limit and suspended
every schedule in-file for exactly this reason. Spending minutes is an owner
call, so the repair is **gated**, not silently taken.

Costs are per main push, measured: katgpt-rs ~119s, riir-chain ~15s (already
paid), riir-ai ~37s, riir-neuron-db ~6s. The two new ones are cheap; katgpt-rs
is the expensive one because its `.lake` is Mathlib-backed.

**Owner decision needed — pick one:**
- (a) wire all three into their `lean_proofs.yml` jobs (~162s/main push total), or
- (b) wire only the two cheap ones (riir-ai + riir-neuron-db, ~43s) and leave
      katgpt-rs to the workstation, or
- (c) leave CI as-is and rely on the workstation run, or
- (d) make them `workflow_dispatch`-only, so each is at least nameable without
      firing on every main push.

## Incidental — a numbering observation that was WRONG, and why

⛔ **Retracted 2026-09-12.** The original text read: `.highwater` says 747 and
the max issue file ever added is 747, yet AGENTS.md cites Issue 750 as this
repo's — so "either 750 was allocated without its file ever being committed, or
the allocator regressed." Both hypotheses were false, and the allocator was
correct the whole time.

What was actually true: the citation was **riir-ai's** Issue 750
(`750_behavior_first_quantization_promotion_gate.md`, the lossy-surface
promotion rule), written WITHOUT naming its repo. It has since been qualified
to "riir-ai Issue 750" in both AGENTS.md and HISTORY.md. katgpt-rs has since
allocated its own, unrelated `750_checks_array_vs_its_own_documentation.md`.
Two live 750s in two repos, exactly as designed — numbers are per-repo.

**This is Issue 749's failure mode, observed happening to a reader.** 749 says
an unqualified cross-repo `Issue N` citation "rebinds to the WRONG document
once that number is allocated locally." That is not a hypothetical: a reader
(this issue's author) saw a bare `Issue 750`, resolved it against the local
repo because nothing said otherwise, found no local file, and inferred an
allocator defect that did not exist. The rebinding cost a wrong diagnosis
written into a tracked document — cheap here, because the conclusion was
recorded as an observation rather than acted on.

The general lesson is narrower than "check your numbers": **a missing repo
qualifier does not read as missing.** It reads as a local reference, silently
and with full confidence, because "local" is the reader's default. That is why
`issue_citation_gate.py` had to exist — and it now PASSES, every cross-repo
citation naming its repo.

748 remains correct for this issue: it was uncontested under every source of
evidence when taken, and `numbering_gate.py` is green.

## Related

- Layer-5 wrap repair, ported to all four this session: katgpt-rs `ad7fc9a7`,
  riir-ai `044445651`, riir-chain `cbe76d3d`. Origin riir-neuron-db Issue 605.
- Negative tests added to riir-ai (`8b0de0f64`) and riir-neuron-db (`41f3dcd`)
  on 2026-09-12, closing the "one-sided gate" half of this issue. Every Lean
  gate in the workspace is now proven non-inert.

Two things the new instruments found immediately, which is the argument for
having built them:

- **riir-neuron-db Issue 617** (`3a2811e`): two layout `_eq_sum` theorems
  restate their own definitions and cannot fail on a wrong constant. Not a
  crisis — the concrete-literal siblings carry the weight — but the doc prose
  around them claims coverage they do not provide.
- **riir-ai `0a6c02f57`**: a retraction. The negative test shipped with a scope
  note asserting a proof hole (a fear-sign flip "builds GREEN"). The experiment
  had never been run; measured, the flip FAILS at `Hla/SpecTests.lean:53` and
  `:65`. The note now states the real residual — three point-examples, one of
  them degenerate, give only two independent constraints on five coefficients.

  The lesson is the same one this issue is about, one level up: an *asserted*
  outcome reads exactly like a measured one. A claim that an experiment would
  come out a certain way is not evidence, even when the reasoning is plausible.

## Both follow-ups CLOSED (2026-09-12, later the same day)

Neither needed an owner call — both were coverage gaps with a mechanical fix,
and both turned an *argued* claim into a *measured* one.

### riir-neuron-db `c630654` — 9/9 becomes 12/12

`Construction/Layout.lean` was the one spec module with **no** perturbation.
Nine rejections over four files read as a suite-wide verdict and were not one:
a file nothing perturbs contributes no evidence, and its absence is invisible
in the pass count. Three added, each breaking a different theorem (variant tag
collision → `variant_tags_strictly_increasing`; `envelopeRefSize` 32→4 →
`sidecarHeaderSize_eq`; `n_facts` offset dropping `variantSize` →
`offset_chain_monotone`).

**Issue 617 is now measured rather than read, and the measurement narrows it.**
The wrong-constant perturbation reds at the LITERAL theorem `:123` and never at
the vacuous `_eq_sum` `:120` — the finding confirmed. But removing the literal
theorem *still* reds, in `Construction/SpecTests.lean:82,97,98` (control with
correct constants is green, so the probe is valid). So the constant has **two**
independent sound sites and `_eq_sum` is neither: redundant, not load-bearing.
That is what the owner's style call was missing.

### riir-ai `69cc69f4e` — the residual gap CLOSED, not just restated

`0a6c02f57` left the residual as prose: "a three-parameter family of wrong
weight vectors still passes." That family is no longer described, it is
**constructed**. Solving the two constraints for a change confined to
(arousal, valence, calm) gives `dv = -2dc`, `da = dc`; at `dc = 0.01` the
weights `0.31 / 0.23 / 0.21 / -0.15 / -0.10` change the `oneScalars` sum by
**exactly 0** and the `testScalars` value by **exactly 0**.

Measured two-sided: with only the original three examples that wrong spec
**builds GREEN**; with five new one-hot examples it reds at exactly the three
perturbed weights. It is now perturbation **[9/9]**, so the closure cannot
silently regress.

⛔ **The planned repair was the wrong one, and cost more.** This issue recorded
that closing the gap "needs a monotonicity/antitonicity theorem, which would
raise `EXPECTED_THEOREMS` past 16". A monotonicity theorem pins the five
**signs and no magnitude** — strictly weaker than what shipped. Five one-hot
instances give five independent equations, one per coordinate, pinning every
weight exactly; and because `example`s are anonymous, the audited surface is
**unchanged at 16** (`proof_gate.sh` counts `#print axioms` directives). The
blocker in the plan was an artifact of the chosen mechanism, not of the goal.

### A harness gap found while doing both, fixed in both

Both scripts' headers require every perturbation to break because a **proof**
no longer holds, "never because the file stopped parsing" — and **nothing
asserted it**. A perturbation that merely mangled a file would have been
counted as a rejection while the hole it was meant to probe stayed open.
`perturb()` now discriminates and reports a syntax break as a
`PERTURBATION BUG`; probed on a COPY of the harness with planted defects
(editing a running shell script changes its own execution), all three verdicts
distinguishable.

⛔ And the first version of that arm **aborted the run after [1/12] while the
pass count read normal**: under `set -e`, a bare `out="$(failing cmd)"`
assignment returns the command's status, so errexit killed the script at the
first perturbation that did its job. `|| rc=$?` is what makes the assignment a
tested command. A harness repair is a harness change, and needs its own probe.

**Still owner-gated:** wiring any of this into CI (the four costed options
above). Three negative tests now exist that nothing invokes automatically —
the gap this issue names is unchanged in kind, and one perturbation wider.
