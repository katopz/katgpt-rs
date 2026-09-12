# Issue 753 — the citation rules' own COSTS were recorded once and never re-measured

**Status:** RESOLVED 2026-09-12 — both costs now re-measured every run, both
probes proven to fire, the width class pinned at 0 with a two-sided revert
probe. One prose row repaired in katgpt-web (its ceiling lowered in the same
commit).

## The class

`issue_citation_gate.py` / `citation_drift_sweep.py` (Issues 749-752) carry two
deliberate narrowings. Each was measured ONCE, written into a docstring, and
then read forever as a fact:

| rule | recorded as | where |
|---|---|---|
| citation width `\d{2,4}` | "recorded blind spot, measured at zero cost today: **0** single-digit citations in the workspace" | 751 T2b |
| alias reach = 40-char LEAD | "aliases restricted to the 40-char lead" — listed as an unrepaired blind spot, never costed | 752 handoff |

Both are claims about a corpus **five-plus concurrent sessions edit daily**.
This module's own governing lesson applies to its own documentation: a count
that matches is still a claim, not a checksum. And a blind spot recorded as
*cost-free* invites the wrong repair — "widen `\d{2,4}` to `\d{1,4}`" reads as
free money to the next agent who opens the file.

## What was measured (2026-09-12)

### (a) The width bound is LOAD-BEARING, not a blind spot awaiting repair

Measured over **every tracked `.md` in 19 contract repos** (5,043-5,057 files;
the corpus moved during the session, for the ordinary reason):

| | pinned scope (AGENTS.md + HISTORY.md, 35 docs) | all tracked `.md` |
|---|---|---|
| single-digit HEADS `\d{1,4}` would add | **0** | **51** |
| single-digit list TAILS it would add | **0** | **0** |

All 51 are section **numbering inside a document** — `## Bench 1: Throughput`,
`| Bench 3 | … |` — and **0** are citations of `.benchmarks/001_*`. So widening
buys 51 false rows for 0 true ones. The bound stays.

⛔ **A correction of this session's own first number.** The tail figure was
first measured at **55**, with the list expander's **plural precondition
dropped** — `_HEAD` only expands a list when the kind word is plural, so
`Plan 460, 31.5%` and `Issue 096, 2,294 LOC` (the exact measurement-as-citation
shapes that motivated the probe) are inert **today** because both heads are
singular. Costing the width rule alone, with the plural rule held out,
over-stated the cost of widening by 55 — in the direction that flattered the
conclusion I was reaching. The two rules are not independent and neither may be
costed alone. Re-measured with `\d{2,4}` in force: **0** tail expansions
anywhere in the corpus land on the integer part of a measurement.

The head count also moved **44 → 51** between two runs an hour apart. Read it
as a magnitude, per the corpus-moves rule the floors file already states.

### (b) Alias reach: widening FORWARD buys 0 repairs and hides a true finding

The full directory name is accepted from the whole 3-line window — which
includes the citation's own line, forward text and all. Only the SHORT alias is
one-directional. Measured over the CROSS set: **1 of 274** rows has an owner's
alias trailing within 40 chars, and reading it settles the question:

```
riir-game-sdk/HISTORY.md:51   Plan 32 -> owners include riir-chain
  showcase) + `chain_viz` (Plan 032, DeFi dashboard). The chain viz is
```

`chain` there is prose about the **`chain_viz` crate**, not an attribution to
riir-chain. Widening forward would SUPPRESS a genuine finding — itself a
crate-hint row, the class Issue 751 T2(a) ruled must be counted rather than
excused. This is the backward-only window's argument (752) on a second axis,
landing the same way for the same reason: an emitted false positive is read and
dismissed; a suppressed row is invisible to the sample that measures the error
rate. **Lead-only stays — MEASURED, not assumed.**

## The repair — a dated zero becomes a per-run assertion

1. `unseen_by_width(text)` and `alias_trail_owners(line, kind, n, sibs)` live in
   `issue_citation_gate.py` next to the rules they cost, and both consumers
   share them (DRY — the sweep re-implements nothing).
2. The **gate** counts the width complement every run and pins it:
   `max_single_digit = 0`. A breach is **exit 2 (instrument untrustworthy)**,
   not exit 1 (prose drift) — the prose did not get worse, the scope grew a
   form the verdict regex cannot read, so a clean verdict stopped covering it.
3. The **sweep** reports both quantities workspace-wide beside the verdict, with
   the same standing as tail support and the adjudication count: they ORDER
   work and size a decision's cost; they are never a second verdict.
4. **Both probes are proven to fire.** Sweep selftest: a planted `## Bench 1:`
   raises the width count to 1 *and must not enter the citation walk*; a
   3-digit citation must not raise it; a trailing `fakesib` alias must still
   produce a CROSS row *and* count as a suppression cost; the same alias in the
   LEAD must qualify (no row, no cost); a trailing NON-owner alias must not
   count. Gate: planting `Issue 6` in AGENTS.md → **exit 2**, restore →
   **exit 0**, AGENTS.md byte-clean after.

## The katgpt-web adjudication (its Issue 003, the last blocked row)

`katgpt-web/AGENTS.md:29` cited a bare `Research 003` while the same paragraph
forbids ever *naming* a private `riir-*` repo. Seven repos own the number, of
which one is public. Decided on evidence, not permission:

- katgpt-rs's `.research/003_Commercial_Open_Source_Strategy_Verdict.md` is the
  document that **defines the axis the sentence invokes** — "let public-research
  agents self-govern the public/private boundary without needing the sensitive
  moat doc", "the table above is the public/private axis".
- riir-ai carries a same-titled copy which **differs**; it is the private
  variant and is inadmissible here regardless.

So the referent is katgpt-rs's, the qualifier is admissible, and the row is
repaired rather than reworded. katgpt-web `cross 1 -> 0`, ceiling lowered in the
same commit.

⚠ One ownership row looked like an instrument bug and was not: seal-game-editor
has **no `.research/` directory** yet is reported as owning Research 3. It
allocated `.research/003_migration_gap_audit.md` and removed it under the
noise-reduction rule; `allocated()` walks `git log --all` precisely so a
removed-but-allocated number still counts. Checked, not assumed.

## Standing after this issue

Both narrowings are now decisions with a re-measured price tag rather than
remembered zeros. The remaining recorded blind spot of the family is
**AMBIGUOUS** (912 rows) — undecidable from the number alone and deliberately
never gated.
