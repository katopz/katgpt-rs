# Issue 752 — a repo name QUALIFIES a citation even when that repo does not own the number

**Status:** RESOLVED 2026-09-12 — repair landed, revert-probed, floors re-pinned

`issue_citation_gate.py` and `citation_drift_sweep.py` decide a cross-repo
citation is followable when a sibling repo's name appears in the citation's
3-line window. The predicate asks **"is a repo named?"** and never **"does
that repo own the number?"** — so a citation whose prose names the *wrong*
repo reads as clean, and one whose window merely happens to contain an
unrelated repo name reads as clean too.

This is the exact argument Issue 751 T2(a) already made about **crate**
hints, and won: a plausible address that is wrong is worse than no address,
so crate hints are sub-labelled `⛔MISLEADING` and still **counted as
findings**. The directory-name path was exempted from that argument with no
measurement behind the exemption — and it is the path that *silently
absolves*, where the crate path only annotates.

## Measured (2026-09-12, 19 contract repos / 2,940 citations / 35 documents)

Of **368** citations the current rule QUALIFIES:

| bucket | n | what it is |
|---|---|---|
| owner-consistent | 323 | a named repo really does own the number — correct |
| `WINDOW_ONLY` | 37 | the 3-line window holds a sibling name that is not an attribution at all (a crate-inventory table row, an adjacent unrelated clause) |
| `ADJACENT` | 8 | an explicit attribution sits ON the citation and names a repo that does **not** own the number |

**All 45 are citations the reader cannot follow** — no repo named beside them
owns the cited number. They are false NEGATIVES of CROSS, and folding them in
moves the corpus figure **254 → ~299**, a **~15% under-count**. That is the
same magnitude as the sweep's measured **16% false-positive** rate and in the
opposite direction; neither was known when 254 was first quoted, and the
number is not quotable without both.

### The 8 ADJACENT rows, hand-adjudicated

**3 are wrong addresses** — follow the prose, arrive at a repo without the number:

- `riir-deployer/HISTORY.md:19` — "`riir-chaind roll`, **riir-chain Plan 211**".
  riir-chain's `.plans` tops out at **058** (`.highwater` = 58). Plan 211 is
  katgpt-rs/riir-ai's.
- `riir-game-sdk/HISTORY.md:769` — "**riir-mmorpg-examples Issue 059**".
  riir-mmorpg-examples allocated 001–030, 032–058, 061… and **never 059**
  (worktree and `git log --all` both). Owners: katgpt-rs, riir-chain,
  riir-clippy, seal-game-editor.
- `riir-neuron-db/AGENTS.md:82` — "**katgpt-rs Issue 513** T6". 513 is
  riir-ai/riir-train's; katgpt-rs owns the *script* the line is about, not the
  issue. The most instructive row: the attribution follows the **code**, and
  the number follows the **document**.

**2 are borderline** — `riir-viewbridge/AGENTS.md:56,456`, where `seal-remake`
attaches to a crate (`seal-remake`'s `seal-node`) in a clause beside `Plan 031`.

**3 are classifier false positives** — the adjacent token is not an
attribution: `dep:riir-auth` (a cargo dependency expression, HISTORY.md:674),
a semicolon clause boundary (HISTORY.md:734), and the alias `chain` matching
inside **"chain-validation"** (riir-mmorpg-examples/HISTORY.md:295). That last
one is its own sub-finding: the gate's alias table warns in a comment that
`chain`/`train`/`shader` are ordinary words and restricts aliases to the
40-char lead — **the 40-char lead is not enough**, and owner-consistency
fixes it incidentally, because an ordinary word that happens to name a repo
almost never names the *owning* repo.

So the sub-label is a **hint that ORDERS the repair, never a verdict** —
identical standing to `⛔MISLEADING` and to tail support in the percentile
audit. 3/8 adjudicated as genuinely wrong addresses; all 8 unqualified.

## Repair

Qualification becomes **owner-consistent**: a named repo qualifies a citation
only when that repo is in the cited number's owner set.

Owner-consistency applies **exactly when the number has owners**. When no repo
in the workspace owns it (`ORPHAN`), there is nothing to be consistent with,
and any named repo is accepted — the row is a dangling reference, a different
repair from a rebinding hazard, and it keeps its own bucket.

Rows that lose their qualification fall through to the existing classifier and
land in `CROSS` / `IN-LOCAL-RANGE` / `ORPHAN` on their merits. A CROSS row
carrying an adjacent non-owner attribution is sub-labelled `⛔MISATTRIBUTED`.

## Tasks

- [x] T1 — shared `qualifiers()` in `issue_citation_gate.py`; sweep reuses it verbatim (it already imports `citations`/`aliases`/`allocated`/`KINDS`)
- [x] T2 — owner-consistency in gate + sweep; `⛔MISATTRIBUTED` sub-label
- [x] T3 — re-pin `issue_citation_floors.txt` + `citation_drift_floors.txt`; the sweep's self-assertion against the gate must still AGREE
- [x] T4 — revert probe: a planted non-owner attribution must RED; restore byte-clean
- [x] T5 — record the corrected corpus figure + both error rates in the sweep docstring and AGENTS.md

## Why this could not be found by reading the code

The rule is three lines and reads as obviously correct — "a repo name in the
window makes the citation followable". It is wrong only against the
**allocation walk**, which the same file already computes for a different
purpose. The defect is not in either half; it is that the two halves were
never joined.

## Landed

`254 -> 291` CROSS, `46 -> 54` IN-LOCAL-RANGE, `0` ORPHAN unchanged; 7
`⛔MISATTRIBUTED` rows now labelled. Ten per-repo ceilings re-pinned in the
same commit (riir-ai 1→2, riir-auth 6→8, riir-clippy 32→36, riir-dao 5→6,
riir-dapps 18→21, riir-deployer 7→10, riir-game-sdk 108→124,
riir-mmorpg-examples 41→43, riir-neuron-db in-range 1→3, riir-viewbridge
12→17) — a ceiling that RISES on an instrument repair is the honest direction,
and the floors header now says so, because a ceiling that FALLS is ambiguous
between a repair and a blinded classifier.

**katgpt-rs itself stayed at 0 findings**, which is why the gate could be
tightened here at no cost — the ideal moment to tighten one.

**T4 revert probe.** Planted `riir-clippy Issue 513` (riir-clippy does not own
513) in AGENTS.md; the gate RED with the right label —
`⛔MISATTRIBUTED — names riir-clippy, which does NOT own 513` — and AGENTS.md
restored byte-clean (`git diff --quiet` YES). The probe is two-sided: the OLD
rule passed that exact line, because a repo WAS named.

**A second-order check that landed in the same pass.** Adding three citations
to AGENTS.md's own prose about this repair moved the walk 2,945 -> 2,948, which
would have made the hand-typed figure in AGENTS.md drift within the hour. The
prose now carries the MAGNITUDE (~2.9k) and the sweep's docstring the dated
snapshot — a figure five-plus concurrent sessions can move does not belong in
a document as if it were a checksum.
