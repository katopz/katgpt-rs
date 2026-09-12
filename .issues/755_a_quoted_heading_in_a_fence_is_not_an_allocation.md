# Issue 755 — a heading inside a fenced code block reads as an ALLOCATION, and the allocation path is the one that suppresses findings

STATUS: RESOLVED 2026-09-12 — `fenced_lines()` landed on the allocation path only; the asymmetry measured on BOTH sides (0 of 57 headings fenced, 57 of 2972 citations fenced); 3 selftest arms canaried two ways; unterminated fences fail SAFE and RED

## The defect

Issue 754 recorded its own known gap and asked for it to be re-measured before
being relied on:

> **Known gap, measured empty:** a heading inside a fenced code block is not
> excluded by fence tracking […] The one such heading in the workspace fails
> the other two filters anyway.

The gap matters more than "one heading that fails anyway" suggests, because of
which direction it fails in. `heading_allocated()` is the **only** path in this
instrument that can SUPPRESS a finding — a number it reports as locally
allocated is a number no citation of which will ever be flagged. A heading
inside a fence is a **quoted example**, typically of *another* repo's document,
and these documents quote headings constantly: that is what the prose is for.
Accepting one does not merely drop a row, it **inverts** one — the exact defect
754 was filed to repair, re-introduced through a different door.

## The measurement — both sides, because the two surfaces need OPPOSITE rules

Measured over the pinned documents (AGENTS.md + HISTORY.md) in all 19 contract
repos, 2026-09-12:

| surface | population | fenced | rule |
|---|---|---|---|
| `heading_allocated()` — allocations | 57 `_SELF_HEADING` matches | **0** | EXCLUDE fences |
| `citations()` — findings | 2972 citations | **57 (1.9%)** | do NOT exclude |

The second row is why this is an asymmetry rather than a scanner-wide repair,
and it was nearly landed the wrong way. Those 57 are not incidental: the
fenced rows include riir-auth's sibling-layout block —

```
/git/riir-ai          ← guard.wasm (ban runtime, Plan 307), seal consumers
/git/seal-online-remaster ← consumer (seal-container-service, Plan 043 Phase 3)
```

— and riir-clippy's equivalent carrying `Bench 010`, `Plan 005` and `Issue
081`. Those are **genuine cross-repo attributions**, in the single place in
these documents where sibling ownership is written down most explicitly.
Excluding fences from the citation walk would have hidden 1.9% of the corpus,
concentrated exactly where the attributions live. The allocation path excludes;
the citation path does not; neither rule was assumed.

⚠ The first read of that 57 was reported to myself as *phantom* citations —
`Plan 5`, `Plan 1`, `Bench 10` against mangled ASCII-art context. Both halves
were artifacts of the sampler: the tuple carries the **parsed int** (`Plan 005`
prints as `Plan 5`) and the context window slices mid-diagram. The rows were
real and correct the whole time. A sampler's formatting is not the data.

## Why not a boolean toggle

A scanner that flips a boolean on every ` ``` ` line mis-phases permanently
after the first **unterminated** fence and thereafter scans the complement —
prose read as code, code read as prose — reporting clean either way
(precedent: `.agents/skills/rust-optimize/SKILL.md`'s unclosed ` ```text `,
43 lines swallowed, which silently ate a new gate's first canary). Matching is
CommonMark-ish instead: the opening run's **char and length** are recorded, and
only a **bare** run of at least that length of the same char closes it, so an
inner ` ```bash ` does not close the block.

That is not theory here — it is a measured arm. Substituting the naive toggle
for `fenced_lines()` returns `[42, 44, 47, 49]` against an expected
`[42, 45, 49]`: it **drops a real allocation** and **admits two quoted ones**
in the same pass. Two errors, in opposite directions, from one inversion.

## Unterminated fences fail SAFE, and RED

An unterminated fence would otherwise exclude its tail to EOF — which is the
suppression this filter exists to prevent, merely relocated past the point
anybody would look for it. So `fenced_lines()` returns an **empty** exclusion
set together with the opening index: nothing is excluded, behaviour is exactly
today's measured-correct behaviour, and the hazard is surfaced separately by
`unterminated_fences()`. `issue_citation_gate.main()` exits **2** (instrument
untrustworthy, not prose drift) on one in any contract repo's pinned documents,
because it is both this filter's premise and a real rendering bug.

Measured at landing: **0** unterminated fences across the 35 scoped documents.

## Liveness — 3 arms, canaried two ways

`citation_drift_sweep.selftest()` gains a fixture repo whose HISTORY.md carries
two real allocations and four quoted ones, arranged so the two CommonMark
nuances are each load-bearing: an **info string** (` ```bash `) must not close
a block, and a **shorter** run must not close a longer opening. Its AGENTS.md
carries an unterminated fence, asserting all three of: empty exclusion set,
reported opening index, and that the heading after it is still allocated.

Canaried both ways, because a pin that cannot fail certifies nothing:

| disarmament | result |
|---|---|
| filter removed (exclude nothing) | 3 arms red — `[42, 43, 44, 45, 47, 48, 49]` |
| naive in/out toggle | 4 arms red — `[42, 44, 47, 49]`, wrong in both directions |
| restored | green |

⛔ The fixture itself was wrong first, and the selftest caught it: it asserted
that a longer bare run (` ```` `) does **not** close a ` ``` ` block. CommonMark
says it does, so the block re-opened on the next line and ran unterminated to
EOF — which correctly disarmed the filter and red the arm. The corrected
fixture pins the rule that actually holds (a **shorter** run cannot close a
longer opening). A selftest whose fixture encodes a wrong premise fails in the
same direction as no selftest at all; this one only survived because the arm
red on the first run rather than the tenth.

## Verdict-affecting? No — and that is the point

0 of 57 headings are fenced today, so no verdict moves: sweep still **3 CROSS ·
0 ⛔MISATTRIBUTED** over 2972 citations, gate still green at 152 citations,
docs gate 16/16. This lands for the direction it closes, not for the rows it
retires. Per Issue 754's own lesson — a false allocation does not drop a row,
it inverts one — a suppression path measured empty is worth a filter and a pin,
because the day it stops being empty it stops being visible.
