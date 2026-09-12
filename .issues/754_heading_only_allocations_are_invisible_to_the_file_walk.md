# Issue 754 — an allocation that exists only as a HEADING is invisible to `allocated()`, and the census that certified the rule inherited the blindness
STATUS: RESOLVED 2026-09-12 — `heading_allocated()` landed with a four-arm two-sided selftest; 7 numbers recovered workspace-wide, 123 CROSS rows retired, Issue 752's census re-rated 0/45 -> 1/45

## The defect

`issue_citation_gate.allocated(repo, subdir)` answers "which numbers has this
repo ever allocated?" from two sources: the worktree walk of `.issues/` (etc.)
and a `git log --all --name-only` walk of the same path. History is already
there deliberately — the noise-reduction rule REMOVES a resolved issue file, so
a worktree-only walk reports a live citation as dangling.

**Both walks see nothing when a file is created and removed without an
intervening commit.** That is not hypothetical: it is the *normal* shape for a
same-day issue in the fast-moving consumer repos, where `## Issue 076
(2026-08-19, RESOLVED same day)` is a routine HISTORY.md heading. The repo's
own heading is then the **whole** allocation record.

The consequence is not a missing row — it is an **inverted** one. Issue 752
made ownership load-bearing: a repo name qualifies a citation only if that repo
OWNS the number. Feed that rule a blind `allocated()` and correct prose gets a
`⛔MISATTRIBUTED` verdict, the sweep's most severe:

    riir-game-sdk HISTORY.md:769
      "`demo_coverage_curiosity` (riir-mmorpg-examples Issue 059, 2026-08-14)"
      -> ⛔MISATTRIBUTED: names riir-mmorpg-examples, which does NOT own 59

riir-mmorpg-examples' own HISTORY.md line 1441 reads `## Issue 059
(2026-08-14) — Demonstration-teachable pets (BDH-CQ live consumer)`. Same
number, same date, same subject. The prose was right and the instrument was
wrong, and it said so in the vocabulary reserved for the worst class of finding.

## ⛔ What this does to Issue 752's error rate — a census inherits its oracle's blind spots at 100%

Issue 752 reported **0/45** false positives over the rows its ownership rule
recovered, from a **full census** — every row read by hand. That number is now
**1/45**, and the refuted row is one of the three the census singled out as
"outright WRONG addresses": `riir-mmorpg-examples Issue 059`, filed as "that
repo allocated 058 and 061 but never 059".

A census is exhaustive over ROWS, not over the ORACLE it checks each row
against. Reading all 45 could never have caught this, because every read asked
`allocated()` the same question and got the same wrong answer. **`0/45` was a
statement about the reader, never about the rule** — and the two remaining
examples (`riir-chain Plan 211`, `katgpt-rs Issue 513`) are unaffected only
because they happen to fail for a different reason.

This is the third time in this family that a measured error rate bounded only
the direction somebody thought to sample (751's 16% over-count, 752's ~15%
under-count, now this). The standing rule — never quote one rate over two
populations — gains a corollary: **never quote a rate without naming the
instrument the sample was adjudicated against.**

## The repair — `heading_allocated()`, and why its filters are measured

`scripts/issue_citation_gate.py`:

    allocated(repo, subdir) = worktree walk | git-log walk | heading walk

The heading walk is the only path in this instrument that can **SUPPRESS** a
finding, so its predicate was measured against every candidate in the
workspace, not reasoned about. 11 headings in 19 repos match `^#+ <Kind> <N>`
and are absent from the file walks; **7** are real local allocations and 4 are
not. Two filters separate them exactly:

| rejected | why |
|---|---|
| `## Issue 667 consumer-side follow-ups (2026-08-14/15)` | words between the number and the parenthetical — a heading ABOUT a foreign number. Same shape: `## Plan 539 follow-up (…)` |
| `## Issue 092 (riir-mmorpg-examples) — …` | the parenthetical NAMES the owner, and it is not this repo |
| `# Proposal 031 §0 + §3: …` | H1 (a document title), and inside a fenced code block besides |

So the accepted form is `^##+ <Kind> <N> (<parenthetical naming no other
contract repo>)`. Recovered: riir-mmorpg-examples 059/060/076/077/078/083 and
seal-remake 015 — all `.issues`, all same-day-resolved.

**Known gap, measured empty:** a heading inside a fenced code block is not
excluded by fence tracking (a naive in/out toggle inverts on the first
unbalanced fence and then scans the complement). The one such heading in the
workspace fails the other two filters anyway. Re-measure before relying on it.

**Not extended to inline bold.** riir-game-sdk's `**Issue 005 extension
(2026-07-19):**` is the same class (its own number, filed and removed same-day)
and is deliberately NOT recovered: inline bold is common enough in this prose
that accepting it would absolve real wrong addresses. Those rows stay
IN-LOCAL-RANGE, which is the honest verdict for them.

## Liveness — four arms, canaried both ways

`citation_drift_sweep.selftest()` builds a fixture repo carrying one acceptable
heading and all three measured rejects, and asserts: the positive fires, each
negative does not, the KIND is read from the subdir (a `Plan` heading must not
land in `.issues`), and `allocated()` actually unions the path.

Verified two-sided, because an arm that cannot fail certifies nothing:

- **disarmed** (`heading_allocated` stubbed to `set()`) -> 3 failures
- **loosened** (`[^(]*` between number and parenthetical, H1 accepted) -> the
  arm reports `[42, 43, 45]` against an expected `[42]` and fails

## Measured effect

| | before | after |
|---|---|---|
| workspace CROSS | 239 | 116 |
| IN-LOCAL-RANGE | 50 | 22 |
| ⛔MISATTRIBUTED | 5 | 2 |
| repos at cross=0 | 9 | 10 (seal-remake joined) |

122 of the 123 retired CROSS rows are riir-game-sdk prose repairs landed the
same day (that repo's `.issues/030`); the rest of the movement — and all of the
IN-LOCAL-RANGE fall — is this instrument change. The two are separated in the
pin file's comments on purpose: **a ceiling that falls is ambiguous between a
repair and a blinded classifier, and only the commit can say which.**

## Re-measure

    python3 scripts/citation_drift_sweep.py --full
    ./scripts/docs_gate.sh
