#!/usr/bin/env python3
"""GATE: a CROSS-REPO `Issue N` / `Plan N` / `Bench N` citation that names no repo.

`numbering_gate.py` protects a number from being allocated TWICE in one repo.
It structurally cannot see the failure one level up: a citation whose referent
lives in a DIFFERENT repo. `Issue 750` in this repo's prose meant
`riir-ai/.issues/750_behavior_first_quantization_promotion_gate.md`, and this
repo's own `.issues/.highwater` read **748** — two away. The moment anybody
follows the Numbering Discipline and allocates 749, then 750, that citation
stops dangling and starts resolving, silently, to the WRONG document. A
dangling reference is an inconvenience; a reference that rebinds to a real but
unrelated issue is a wrong answer delivered with a straight face.

Measured when this gate landed (2026-09-12): EIGHT unqualified rows over THREE
numbers — `Issue 513` (riir-train, x3), `Issue 750` (riir-ai, x3), `Issues
490/493` (riir-ai, x2). Every one resolved uniquely by TITLE match, and the
convention for writing them was already in the same file four lines away
("riir-ai `.issues/892`", "894 resolved same day in riir-ai").

⛔ **The AMBIGUOUS bucket is this gate's stated blind spot, not a clean pass.**
A number that exists BOTH locally and in a sibling is undecidable from the
number alone, and there were **35** such cited Issue numbers at landing. This
gate is green over them by construction: its predicate is "does NOT resolve
locally", so a bare `Issue 47` naming riir-ai's 47 reads as this repo's 47 and
always will. It is REPORTED every run and deliberately NOT gated: a
ceiling on it reds whenever somebody writes a perfectly correct citation to a
NEW local number that a sibling also happens to have — a nuisance red for a
right action, and a gate that reds on right actions is a gate that gets
bypassed (`staged_set_audit.py`'s rationale, one axis over). So it has the
standing of tail support in the percentile audit: a quantity that ORDERS the
rows and sizes the blind spot, never a verdict. The only real defence is the
writing convention — name the repo whenever the referent is not local,
ambiguous or not.

Exit 0 clean, 1 on an unqualified citation, **2 if the instrument is
untrustworthy** (a floor breached => the walk went blind; a scanned document
missing). An unreliable instrument is not the same finding as drift.

⛔ **The CI lane cannot adjudicate and does not pretend to.** docs_gate.yml
runs per-push on main in a SINGLE checkout — no sibling workspace, so the
ownership lookup is structurally empty and every cross-repo citation would
read as a false finding. A blind run therefore refuses (exit 2) UNLESS the
workflow has marked the context with DOCS_GATE_CI=1, in which case the gate
verifies only the axes decidable from this checkout (documents present, the
width bound, the citation walk over its floor) and says in its LAST line —
docs_gate.sh prints only tail -1 of a passing check — that the cross-repo
axis is DEFERRED to the workstation run. The marker is an explicit opt-in
recorded in the workflow file, never auto-detected, so a blind WORKSTATION
run (a moved repo, a missing sibling) still refuses. Found the hard way:
the gate landed during a main-only CI window and the first promote push was
its first CI run (HISTORY.md 2026-09-12).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from numbering_drift_sweep import contract_repos  # noqa: E402 — the SEVENTH predicate, reused not re-derived
from skill_repo_set_gate import PARTIAL_MARKER, fenced_blocks, partial_clone_state  # noqa: E402 — the fence scanner + the partial-clone axis (Issue 765), reused not re-derived

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

REPO_ROOT = HERE.parent
# Overridable for testing (the skill_repo_set_gate precedent): the
# partial-clone sims (Issue 765) run the real instruments over a symlink
# farm without copying repos.
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", str(REPO_ROOT.parent)))
FLOORS = HERE / "issue_citation_floors.txt"

# ── vocabulary: DATA, not derived ──────────────────────────────────────────
# Deriving both the scope and the population from one walk is what makes a gate
# permanently green. Only the population (which repos, which numbers) is
# derived; the citation vocabulary and the scanned documents are pinned.
KINDS = {
    "Issue": ".issues",
    "Plan": ".plans",
    "Research": ".research",
    "Bench": ".benchmarks",
    "Proposal": ".proposals",
}

# A plural kind word may head a LIST: "Issues 724, 725", "Issues 490/493",
# "Plans 596 and 597". Reading only the head number under-reports the class —
# `Issues 490/493` hid its second referent from the first version of this scan.
# The kind word is CASE-INSENSITIVE (Issue 924, measured 2026-10-07): a house
# style that writes lowercase prose — seal-game-editor is lowercase-DOMINANT
# (`plan` 1,119 vs `Plan` 591; `issue` 719 vs 228 over its tracked text files)
# — was invisible to a case-sensitive head, and the citations this gate counts
# are no exception. `citations()` normalizes the matched word back to the
# canonical KIND so every downstream bucket keys on one spelling.
_HEAD = re.compile(r"\b((?i:%s))(s?)\s+(\d{2,4})" % "|".join(KINDS))
_CANON = {k.lower(): k for k in KINDS}
# `.match(s, pos)` already anchors AT pos — Python `re` has no `\G`.
_TAIL = re.compile(r"\s*(?:,|/|and|&)\s*(\d{2,4})")

NUMBERED = re.compile(r"^(\d+)_")

# ── the WIDTH BOUND is LOAD-BEARING, not a blind spot awaiting repair ────────
# `\d{2,4}` was recorded (Issue 751 T2b) as "a blind spot, measured at zero
# cost today: 0 single-digit citations in the workspace". True in this gate's
# scope, and the framing invited the wrong repair — widening to `\d{1,4}` reads
# as free. Measured over every tracked `.md` in 19 repos (Issue 753,
# 2026-09-12) it is not free:
#
#   51 false HEADS  — `## Bench 1: Throughput`, `| Bench 3 | …`: a section
#                     NUMBERING inside a document, never a citation of
#                     `.benchmarks/001_*`. 0 of them are real.
#    0 false TAILS  — because the list expander already carries a PLURAL
#                     precondition. That is the load-bearing half, and it is
#                     what keeps `Plan 460, 31.5%` -> Plan 31 and
#                     `Issue 096, 2,294 LOC` -> Issue 2 from ever being built:
#                     both heads are SINGULAR. Re-measured with `\d{2,4}` in
#                     force, 0 tail expansions anywhere in the corpus land on
#                     the integer part of a measurement.
#
# ⛔ The 0 is a correction of this session's own first number, which was 55 —
# measured with the plural precondition DROPPED, and so over-stating the cost
# of a widening in the direction that flattered the conclusion. The two rules
# are not independent and neither may be costed alone. The head count moved
# 44 -> 51 between two runs an hour apart for the ordinary reason (the corpus
# is edited by five-plus concurrent sessions): read it as a magnitude.
#
# So the bound buys ~51 suppressed false rows for 0 suppressed true ones.
# What the class DOES need is liveness: "0 occurrences" is a dated measurement
# over documents edited daily, and the day somebody writes `Issue 6` the
# verdict regex goes silently blind on it. These two patterns are the bound's
# own complement — counted every run, pinned at 0, and a breach is exit 2
# (instrument untrustworthy), not exit 1 (prose drift): the prose did not get
# worse, the scope grew a form the verdict cannot see.
_HEAD_1D = re.compile(r"\b(?i:%s)s?\s+(\d)(?!\d)" % "|".join(KINDS))
# The tail complement carries `_HEAD`'s own PLURAL precondition: a singular
# head never expands a list, which is exactly why `Plan 460, 31.5%` is inert
# TODAY and why dropping the width bound without dropping the plural rule would
# still be a regression. Measuring the complement without it over-states what a
# widening would cost, in the direction that makes the widening look worse.
_TAIL_1D = re.compile(
    r"\b(?i:%s)s\s+\d{2,4}\s*(?:,|/|and|&)\s*(\d)(?!\d)" % "|".join(KINDS))


def unseen_by_width(text: str) -> tuple[int, int]:
    r"""(single-digit heads, single-digit list tails) `\d{2,4}` cannot see."""
    return len(_HEAD_1D.findall(text)), len(_TAIL_1D.findall(text))


def parse_pins(path: Path) -> dict[str, int | list[str]]:
    pins: dict[str, int | list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        pins[key] = [v for v in val.split() if v] if key == "documents" else int(val)
    return pins


_SUBDIR_KIND = {v: k for k, v in KINDS.items()}

# `## Issue 059 (2026-08-14) — Demonstration-teachable pets`: a heading whose
# number is IMMEDIATELY followed by a parenthetical. Both halves are
# load-bearing, and both were measured (Issue 754) against the 11 candidate
# headings in the workspace rather than reasoned about:
#
#   `## Issue 667 consumer-side follow-ups (2026-08-14/15)` — words between the
#       number and the parenthetical: a heading ABOUT a foreign number, not a
#       record of a local one. Same shape: `## Plan 539 follow-up (…)`.
#   `## Issue 092 (riir-mmorpg-examples) — …` — the parenthetical NAMES the
#       owner, and it is not this repo.
#   `# Proposal 031 §0 + §3: …` — H1, and inside a fenced block besides.
#
# 7 of the 11 survive; all 7 read as genuine local allocations. A heading
# inside a fenced code block is EXCLUDED by `fenced_lines()` below — measured
# EMPTY today (0 of 57 matches workspace-wide), so the filter changes no
# verdict and is landed for the SUPPRESSION direction it closes: a quoted
# heading in an example block is not an allocation record, and a false
# allocation does not drop a row, it INVERTS one (Issue 754).
# ── fenced code blocks: excluded from the ALLOCATION path ONLY ──────────────
# The asymmetry is MEASURED on both sides, not reasoned about:
#
#   allocation (`heading_allocated`) — can only ever SUPPRESS a finding, and a
#       heading inside a fence is a QUOTED example, not a record. Excluded.
#       Measured population: 0 of 57 `_SELF_HEADING` matches workspace-wide.
#   citations (`citations`) — PRODUCES the findings, and 57 of 2972 citations
#       (1.9%) live inside fences. They are not incidental: riir-auth's layout
#       block writes `/git/riir-ai  <- ... Plan 307` and riir-clippy's writes
#       `Bench 010` / `Issue 081` — genuine sibling ATTRIBUTIONS a reader
#       follows. Excluding fences there would hide 57 real rows. NOT excluded.
#
# Matching is delegated to `skill_repo_set_gate.fenced_blocks()` — the same
# scanner, imported rather than re-derived, exactly as `contract_repos` is. It
# is already CommonMark-ish (opening run length recorded; only a BARE run of at
# least that length closes it, so an inner ```bash does not) because it was
# written after a naive toggle mis-phased on `rust-optimize/SKILL.md`'s
# unclosed ```text and swallowed that gate's own first canary. A second copy
# here would be a second thing to get wrong.
#
# It reads BACKTICK fences only; `~~~` is unsupported. Measured 2026-09-12: 0
# tilde fences in tracked `.md` across all 19 contract repos (the one hit in
# the workspace is vendored llama.cpp, outside the population). A WATCHED
# class, not a silent one.


def fenced_lines(text: str) -> tuple[set[int], int | None]:
    """(0-indexed lines inside a fence, opening index of an UNTERMINATED one).

    An unterminated fence is returned rather than swallowed: its tail would
    otherwise be excluded to EOF, which is the same suppression this filter
    exists to prevent, just moved somewhere nobody would look for it. Callers
    fail SAFE on it (see `heading_allocated`) and it is surfaced as a verdict.
    """
    inside: set[int] = set()
    open_at: int | None = None
    for first, last, _body, _prev in fenced_blocks(text):
        if last < 0:            # the unterminated-block sentinel
            open_at = first - 1
            continue
        inside.update(range(first - 1, last))
    return (set() if open_at is not None else inside), open_at


def unterminated_fences(repo: Path) -> list[tuple[str, int]]:
    """(document, 1-indexed line) for every unterminated fence in the pinned docs.

    A parse hazard AND a real rendering bug, worth reporting for its own sake:
    katgpt-rs's own rust-optimize SKILL.md once swallowed 43 lines this way.
    """
    out = []
    for doc in _self_docs():
        p = repo / doc
        if not p.is_file():
            continue
        _, open_at = fenced_lines(p.read_text(encoding="utf-8", errors="replace"))
        if open_at is not None:
            out.append((doc, open_at + 1))
    return out


_SELF_HEADING = re.compile(
    r"^#{2,}\s+(?:\*\*)?(%s)\s+0*(\d{2,4})\s*\(([^)\n]*)\)" % "|".join(KINDS))

# Issue 823. The SAME discriminator, at the position it was never applied.
#
# `_SELF_HEADING` is sound because of ONE rule: nothing may sit between the
# number and its delimiter, so `## Issue 043 follow-up (…)` is rejected while
# `## Issue 043 (…)` is read. That rule is about the text AFTER the number.
# It was anchored, silently, to the kind LEADING the heading — and six repos
# write the date first (`## 2026-09-16 — Issue 113: the auto-oracle`), a form
# neither this pattern NOR `_HEADING_SHAPED` could see. Measured: 74 such
# records workspace-wide, 18 of them allocation-shaped under the strict rule.
#
# ⛔ This is NOT the widening AGENTS.md calls unsound, and the distinction is
# the whole justification. That argument is against LOOSENING the
# discriminator — accepting `resolved` / `follow-up`, which no punctuation rule
# separates from an allocation. The discriminator here is IDENTICAL: the number
# must be followed immediately by its title delimiter (`:` or `,`, the
# date-led form's `(`). Measured on the live corpus, it rejects
# `## 2026-09-16 — Issue 152 resolved: …` and `## 2026-09-16 — Plan 064 T3
# landed (…)` exactly as the leading form rejects their siblings. Same
# strictness, one position over.
#
# The foreign-repo filter runs over the WHOLE remainder rather than a
# parenthetical, which is strictly more likely to reject — the safe direction
# for the only path that can SUPPRESS a finding.
_SELF_HEADING_DATED = re.compile(
    r"^#{2,}\s+\d{4}-\d{2}-\d{2}\s*[—–-]+\s*(?:\*\*)?(%s)\s+0*(\d{2,4})\s*[:,(](.*)$"
    % "|".join(KINDS))


# Issue 828. The SAME discriminator, at the DELIMITER it was never spelled.
#
# Issue 823 moved the rule one POSITION over and left it anchored a second
# time — to the delimiter SET: `(` for the leading form, `[:,]` for the dated
# one. The workspace's most common title delimiter is the EM DASH, and it was
# in neither. Measured over the 213 records the oracle declined to read:
# **56** are `## Issue 788 — <title>: CLOSED (date)` — katgpt-rs's own house
# style, its own newest closes, in the repo that owns this instrument. Nothing
# sits between the number and its delimiter in any of them.
#
# ⛔ Not the widening AGENTS.md calls unsound, for Issue 823's own reason: the
# unsound widening is DROPPING the discriminator (accepting `resolved` /
# `follow-up` between the number and the delimiter). This ADDS a delimiter and
# keeps the rule — `## Issue 043 follow-up — title` is rejected here exactly as
# `## Issue 043 follow-up (…)` is rejected by `_SELF_HEADING`, and
# `citation_drift_sweep.selftest()` arm 2 pins the negative in BOTH delimiters.
#
# ⛔ The ASCII hyphen is accepted ONLY space-separated. `## Issue 366-class
# (pos-uniform chunk forward) FIXED in riir-gpu` is a live riir-ai heading
# where the hyphen is part of a WORD, not a delimiter; `-(?=\s)` after `\s+`
# rejects it and an arm pins the case.
#
# group(3) is the WHOLE remainder — `_SELF_HEADING_DATED`'s precedent, not
# `_SELF_HEADING`'s parenthetical. The foreign filter then reads more text,
# which is strictly more likely to REJECT: the safe direction for the only
# path here that can SUPPRESS a finding. `_SELF_HEADING`'s own scope is
# deliberately left alone — widening THAT to the remainder would reject
# `## Issue 059 (date) — <sibling> did X`, which is the suppression Issue 754
# landed, so the safe direction for a new pattern is a regression for an
# existing one.
_SELF_HEADING_DASH = re.compile(
    r"^#{2,}\s+(?:\d{4}-\d{2}-\d{2}\s*[—–-]+\s*)?(?:\*\*)?(%s)"
    r"\s+0*(\d{2,4})\s+(?:[—–]+|-(?=\s))\s*(.*)$" % "|".join(KINDS))


def _self_heading(line: str):
    """(kind, number, scope-text-to-filter-for-foreign-names) or None.

    One matcher, three house styles (Issue 828 added the dash-delimited one).
    Callers must not re-implement the choice: the patterns disagree about which
    group carries the text the foreign filter reads, and that filter is the
    suppression path's only guard.
    """
    m = (_SELF_HEADING.match(line) or _SELF_HEADING_DATED.match(line)
         or _SELF_HEADING_DASH.match(line))
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


_DOCS_CACHE: list[str] | None = None
_NAMES_CACHE: list[str] | None = None


def _self_docs() -> list[str]:
    """The pinned document list, read ONCE from the floors file, not re-typed."""
    global _DOCS_CACHE
    if _DOCS_CACHE is None:
        docs = parse_pins(FLOORS)["documents"]
        assert isinstance(docs, list)
        _DOCS_CACHE = docs
    return _DOCS_CACHE


def heading_allocated(repo: Path, subdir: str,
                      repo_names: list[str] | None = None) -> set[int]:
    """Numbers this repo records for ITSELF in a heading of its own documents.

    A resolved file is REMOVED by the noise-reduction rule; a file created and
    removed without an intervening commit leaves nothing in `git log` either,
    and the repo's own HISTORY.md heading is then the WHOLE allocation record.
    Measured (Issue 754): 7 such numbers workspace-wide, and two of them were
    driving a `⛔MISATTRIBUTED` verdict against prose that was CORRECT —
    riir-game-sdk's `riir-mmorpg-examples Issue 059`, which Issue 752's census
    had adjudicated the other way and recorded as an "outright WRONG address".

    This is the only path here that can SUPPRESS a finding, so its two filters
    are measured (see `_SELF_HEADING`) rather than assumed.
    """
    kind = _SUBDIR_KIND.get(subdir)
    if kind is None:
        return set()
    global _NAMES_CACHE
    if repo_names is None:
        if _NAMES_CACHE is None:
            _NAMES_CACHE = [p.name for p in contract_repos(WORKSPACE)]
        repo_names = _NAMES_CACHE
    foreign = [n for n in repo_names if n != repo.name]
    out: set[int] = set()
    for doc in _self_docs():
        p = repo / doc
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # Unterminated fence => `fenced_lines` returns an EMPTY set, so nothing
        # is excluded and behaviour is exactly today's measured-correct one.
        # The hazard is reported by `unterminated_fences()`, never swallowed.
        fenced, _ = fenced_lines(text)
        for i, line in enumerate(text.splitlines()):
            if i in fenced:
                continue
            hit = _self_heading(line)
            if hit is None or hit[0] != kind:
                continue
            if any(_NAME[n].search(hit[2]) for n in foreign):
                continue
            out.add(hit[1])
    return out


# Issue 781: the SAME shape, without the style anchor. Used only to MEASURE
# what the oracle rejects — never to allocate. Widening the oracle to THIS
# remains unsound and `citation_drift_sweep.selftest()` arm 2 proves it: it
# pins `## Issue 043 follow-up (2026-01-01)` as a measured negative, and
# `043 follow-up (…)` and `097 resolved — … (…)` are the same shape. No
# punctuation rule separates commentary from allocation; the distinction is
# semantic. So the cost is printed instead of guessed at.
#
# ⚠ Issue 823 sharpened what "this" means, and the distinction is load-bearing:
# the unsound widening is DROPPING THE DISCRIMINATOR (accepting any text
# between the number and its delimiter). It is NOT reading a second POSITION.
# `_SELF_HEADING_DATED` keeps the discriminator exactly and moves it to the
# date-led heading, so the negative above is still rejected — by both patterns.
_HEADING_SHAPED = re.compile(
    r"^#{2,}\s+(?:\*\*)?(%s)\s+0*(\d{2,4})\b(.*)$" % "|".join(KINDS))

# Issue 823. The meter was anchored to the same leading position as the oracle,
# so a date-led house style was invisible to the BLINDNESS DETECTOR ITSELF —
# and it failed in the direction that reads as clean. Measured before the fix:
# riir-chain printed `heading_unread=0/1`, a PERFECT score, over 21 records of
# which 20 were unread; riir-dapps printed `0/0` — nothing to measure — over
# 23. A width bound that cannot see a whole house style is not a width bound.
_HEADING_SHAPED_DATED = re.compile(
    r"^#{2,}\s+\d{4}-\d{2}-\d{2}\s*[—–-]+\s*(?:\*\*)?(%s)\s+0*(\d{2,4})\b(.*)$"
    % "|".join(KINDS))


def heading_style_blind(repo: Path, subdir: str,
                        repo_names: list[str] | None = None) -> tuple[int, int]:
    """(accepted, heading_shaped) self-allocation records in this repo's docs.

    A triage quantity with the standing of AMBIGUOUS and the width-bound
    complement — NEVER a verdict, never folded into a finding count. It answers
    one question: how much of this repo's own allocation record does
    `heading_allocated()` decline to read, **on style alone**?

    The foreign-repo filter is applied to BOTH sides, over the whole heading,
    so a row rejected for naming a sibling is not counted as a style loss. The
    gap is therefore exactly the style gap.

    Measured 2026-09-14 over 16 repos x AGENTS.md+HISTORY.md (the
    foreign-filtered population this function counts): **64 of 152 read, 88
    unread**, and the split is by HOUSE STYLE rather than correctness —
    riir-mmorpg-examples 43/43 (`## Issue NNN (date) — title`); riir-ai 0/25,
    riir-clippy 0/25 and riir-train 0/13 (`## Issue NNN resolved — title
    (date)`); mmorpg-remake 14/15; katgpt-rs mixed at 7/29, its own newest closes
    in the form its own instrument cannot read. A dated measurement RECORD, not
    a claim — the live figures are printed by `citation_drift_sweep.py` on
    every run, per repo and in total.

    Why it matters in the direction that is currently 0: an incomplete OWNERS
    set turns a correctly-qualified citation into a FALSE ⛔MISATTRIBUTED —
    Issue 754's exact failure, inherited by Issue 794's MISATTRIBUTED-IN-RANGE.
    Latent, so it is printed rather than remembered.
    """
    accepted = shaped = 0
    for _n, ok in _heading_records(repo, subdir, repo_names):
        shaped += 1
        accepted += 1 if ok else 0
    return (accepted, shaped)


def _heading_records(repo: Path, subdir: str,
                     repo_names: list[str] | None = None):
    """Yield `(number, accepted)` for every heading-shaped self-allocation
    record of this KIND, foreign-filtered.

    ONE walker for both meters. `heading_style_blind` counts it and
    `heading_unread_novel` tests each unread number against the other
    oracles; two copies of this loop is two chances for the width bound and
    the cost bound to disagree about what a record IS.
    """
    kind = _SUBDIR_KIND.get(subdir)
    if kind is None:
        return
    global _NAMES_CACHE
    if repo_names is None:
        if _NAMES_CACHE is None:
            _NAMES_CACHE = [p.name for p in contract_repos(WORKSPACE)]
        repo_names = _NAMES_CACHE
    foreign = [n for n in repo_names if n != repo.name]
    for doc in _self_docs():
        p = repo / doc
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fenced, _ = fenced_lines(text)
        for i, line in enumerate(text.splitlines()):
            if i in fenced:
                continue
            m = _HEADING_SHAPED.match(line) or _HEADING_SHAPED_DATED.match(line)
            if not m or m.group(1) != kind:
                continue
            if any(_NAME[n].search(m.group(3)) for n in foreign):
                continue          # rejected for NAMING a sibling, not on style
            yield int(m.group(2)), _self_heading(line) is not None


def heading_unread_novel(repo: Path, subdir: str,
                         repo_names: list[str] | None = None,
                         known: set[int] | None = None) -> int:
    """Unread records whose number NO OTHER oracle knows - the COST of not
    widening the rule, measured rather than argued (Issue 828 T4).

    Issue 823 T5 left open whether the `## Issue NNN resolved - title (date)`
    family admits a sound discriminator, and both AGENTS.md and
    `citation_drift_sweep.selftest()` arm 2 answer NO: `resolved` and
    `follow-up` are the same SHAPE, no punctuation rule separates commentary
    from allocation, and a false allocation does not drop a row, it INVERTS
    one. That answer is correct and it is not the whole question, because
    nobody had priced it.

    A record whose number is already known from a FILE - in the worktree or
    in `git log` - contributes nothing whichever way the rule goes. Only the
    residue can change a verdict, so the residue IS the blast radius.
    Measured 2026-09-18 over 16 repos: **153 unread, 2 novel** (riir-ai's
    `Issue 969 resolved`, riir-clippy's `Issue 097 resolved`), both exactly
    the Issue-754 never-committed shape. So the 153 is a cost figure that is
    98.7% redundant, and the case for taking an UNSOUND rule to recover it
    does not survive its own arithmetic.

    Printed on every run rather than remembered, because it is the quantity
    that decides the question and it moves whenever a sibling edits a
    heading.
    """
    if known is None:
        known = file_and_history_allocated(repo, subdir)
    return sum(1 for n, acc in _heading_records(repo, subdir, repo_names)
               if not acc and n not in known)


def allocated(repo: Path, subdir: str,
              repo_names: list[str] | None = None) -> set[int]:
    """Every number EVER allocated under `repo/subdir` — worktree AND history.

    History is not optional: the noise-reduction rule REMOVES a resolved issue
    file, so a worktree-only walk reports a live citation as dangling. Issue
    750 is exactly that shape — resolved and removed in riir-ai `b559d2da3`.

    Nor is the FILE walk sufficient (Issue 754): remove a file that was never
    committed and `git log` is empty too. `heading_allocated()` recovers those.
    """
    return (set(heading_allocated(repo, subdir, repo_names))
            | file_and_history_allocated(repo, subdir))


def file_and_history_allocated(repo: Path, subdir: str) -> set[int]:
    """`allocated()` WITHOUT the heading path - the worktree walk + `git log`.

    Split out so the heading oracle's own CONTRIBUTION is measurable rather
    than argued about (Issue 828 T4). Every other member of the union answers
    from a FILE that existed; the heading path exists only for the Issue-754
    shape, where a document was created and removed without an intervening
    commit and its own heading is the whole record.
    """
    out: set[int] = set()
    d = repo / subdir
    if d.is_dir():
        for f in d.iterdir():
            m = NUMBERED.match(f.name)
            if m:
                out.add(int(m.group(1)))
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--all", "--name-only", "--pretty=format:", "--", f"{subdir}/"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    prefix = re.compile(re.escape(subdir) + r"/(\d+)_")
    for line in log.stdout.splitlines():
        m = prefix.match(line.strip())
        if m:
            out.add(int(m.group(1)))
    return out


# Issue 846: the PROSE half of the 842 alias seam. The contract names
# `mmorpg-editor` / `mmorpg-remake` / `mmorpg-remaster` live in repo_set.txt
# only — on BOTH measured boxes (M3, 4090) the directories are on disk as
# `seal-game-editor` / `seal-remake` / `seal-online-remaster`, and every
# document citing their plans was written against the ON-DISK spelling
# (measured: 44 CROSS rows at 846's first real read, nearly every specimen
# naming the owner correctly in prose the matcher could not see). 842's
# `open_repo`/`real()` seam reads the aliased directories; this table lets
# QUALIFICATION accept the spelling as naming the repo, in exactly the two
# places the contract full name is already accepted — on the 40-char lead and
# inside the 3-line window. LENIENCY ONLY: `written_names` (the accusation
# half) stays contract-only, so no new ⛔MISATTRIBUTED class is created and
# a spelling can clear a row but never accuse one.
#
# In CODE, not in the gitignored repo_alias.local.txt: the spellings are the
# same on every box measured, and a box-local qualifier table would make the
# verdict itself machine-local — a floor pinned on this box would mean
# nothing on the next.
DISK_SPELLING_ALIASES = {
    "mmorpg-editor": ["seal-game-editor"],
    "mmorpg-remake": ["seal-remake"],
    "mmorpg-remaster": ["seal-online-remaster"],
    # The 2026-10-01 repo rename (riir-ai/riir-clippy -> riir-ai/riir-refine,
    # plan 192 T1.1): `riir-clippy` is the RETIRED spelling. Every archived
    # citation saying "riir-clippy Issue N" / "riir-clippy Plan N" was written
    # against the old name and must keep clearing — the same leniency-only
    # posture as the on-disk spellings above; `written_names` stays
    # contract-only, so the old spelling can never accuse a row.
    "riir-refine": ["riir-clippy"],
}


def aliases(repo_name: str) -> list[str]:
    """Full directory name, plus a SHORT-FORM alias where one is unambiguous.

    Prose names a sibling both ways — "riir-ai Issue 750" and "dapps Issue 027"
    are equally followable, and a test matching only the directory name calls
    the second one unqualified. Measured over the workspace, 2 of the first 4
    flagged rows were exactly that false positive.

    The alias is the name minus a `riir-` prefix, and ONLY when >= 4 characters:
    `ai`, `kat`, `dao` are too short to appear in prose without colliding with
    ordinary words. Short aliases keep the full name as their only form.

    Deliberately NOT the on-disk spellings (Issue 846): `seal-remake` and
    friends match with the `_NAME` boundary regex — never this function's
    plain `\b`, which matches inside `seal-remake-unity` — and they are
    consumed explicitly by `qualifiers()`. See `spelling_aliases`.
    """
    out = [repo_name]
    stem = repo_name[5:] if repo_name.startswith("riir-") else ""
    if len(stem) >= 4:
        out.append(stem)
    return out


def spelling_aliases(repo_name: str) -> list[str]:
    """The ON-DISK directory spellings a repo is known by (Issue 846).

    The contract names `mmorpg-editor` / `mmorpg-remake` / `mmorpg-remaster`
    live in repo_set.txt only — on BOTH measured boxes (M3, 4090) the
    directories are on disk as `seal-game-editor` / `seal-remake` /
    `seal-online-remaster`, and every document citing their plans was written
    against the ON-DISK spelling (measured: 44 CROSS rows at the first real
    read, nearly every specimen naming the owner correctly in prose the
    matcher could not see). 842's `open_repo`/`real()` seam reads the aliased
    directories; this is the PROSE half of the same seam.

    A spelling qualifies in exactly the two places the contract full name is
    already accepted — on the 40-char lead and inside the 3-line window — and
    matches with the `_NAME` boundary regex, so `seal-remake-unity` (the
    retired repo) does not name `seal-remake`. LENIENCY ONLY:
    `written_names` (the accusation half) stays contract-only, so a spelling
    can clear a row but never accuse one.

    In CODE, not in the gitignored repo_alias.local.txt: the spellings are
    the same on every box measured, and a box-local qualifier table would
    make the verdict itself machine-local — a floor pinned on this box would
    mean nothing on the next.
    """
    return DISK_SPELLING_ALIASES.get(repo_name, [])


class _NameRx(dict):
    """`riir-viewbridge` names `mmorpg-remake-unity`; a plain `"mmorpg-remake" in
    ctx` reads that as naming **mmorpg-remake**, a different repo, and qualified
    a `Plan 031` citation on it (Issue 752). A repo name is only a repo name
    when no further name-segment extends it — `riir-ai/scripts/…` and
    `riir-ai's` still match, `riir-games-mmorpg` does not match
    `riir-game-sdk`."""

    def __missing__(self, name: str) -> re.Pattern:
        rx = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        self[name] = rx
        return rx


_NAME = _NameRx()

# An alias only qualifies a citation when it sits right ON it ("dapps Issue 27"),
# never merely somewhere nearby: `chain`, `train` and `shader` are ordinary
# words in this prose, and a 3-line window full of them would qualify every
# citation in the repo and quietly retire the gate. The FULL directory name is
# unambiguous enough to accept from the wider window.
_ALIAS_REACH = 40


def written_names(lead: str, sibs: list[Path]) -> set[str]:
    """Repos whose FULL directory name is literally written on this citation.

    The accusation half of `qualifiers()`. `adjacent` pools full names with
    SHORT-FORM aliases, and that pooling is safe in only one direction: an
    alias match that QUALIFIES a citation is a leniency (it clears a row), and
    an alias match that ATTRIBUTES one is an accusation — reported as
    `⛔MISATTRIBUTED`, a class walled at 0.

    ⛔ Measured 2026-09-15, and the hazard was already written down one
    function below: *"`chain`, `train` and `shader` are ordinary words in this
    prose."* riir-game-sdk's `AGENTS.md:160` reads *"the Active-preview mirror
    client chain (Plan 199 Phase E A5/E3)"*, and the bare word **chain** — 38
    characters ahead, inside the 40-char lead — produced
    `⛔MISATTRIBUTED: names riir-chain, which does NOT own 199`. The string
    `riir-chain` does not appear in that file at all. One false accusation,
    hard-failing a repo's sweep, on prose that named no repo.

    An accusation that a document wrote the wrong address must be able to QUOTE
    the address. A short alias is not an address somebody wrote.
    """
    return {s.name for s in sibs if _NAME[s.name].search(lead)}


def qualifiers(lines: list[str], ln: int, lead: str, sibs: list[Path]) -> tuple[set[str], set[str]]:
    """Repo names the prose offers as this citation's address -> (window, adjacent).

    `window` is the full directory name anywhere in the 3-line backward window;
    `adjacent` is the subset sitting ON the citation (inside `lead`), plus the
    short-form aliases and the on-disk spellings (Issue 846), which are only
    ever accepted there. The window ALSO accepts an on-disk spelling in place
    of the contract full name — a spelling is a full directory name, written
    against a box where the directory carries that name.

    Split because the two carry different weight once the caller checks
    OWNERSHIP (Issue 752): an adjacent non-owner is somebody writing a wrong
    address, a window-only non-owner is a name that was never an attribution.
    """
    ctx = "\n".join(lines[max(0, ln - 3):ln])

    def _names_in(text: str) -> set[str]:
        """Full names + on-disk spellings, both at `_NAME` boundaries."""
        out = set()
        for s in sibs:
            if _NAME[s.name].search(text):
                out.add(s.name)
            elif any(_NAME[v].search(text)
                     for v in spelling_aliases(s.name)):
                out.add(s.name)
        return out

    window = _names_in(ctx)
    adjacent = written_names(lead, sibs)
    adjacent |= _names_in(lead)
    adjacent |= {s.name for s in sibs for a in aliases(s.name)[1:]
                 if re.search(rf"\b{re.escape(a)}\b", lead)}
    return window | adjacent, adjacent


def alias_trail_owners(line: str, kind: str, n: int,
                       sibs: list[Path]) -> set[str]:
    """Repos whose SHORT alias sits just AFTER the citation — the population
    `qualifiers()` deliberately does not read, reported so the cost of that
    decision is re-measured rather than remembered.

    `_ALIAS_REACH` is a LEAD: aliases qualify only when they precede the number
    ("dapps Issue 27"). The full directory name is accepted from the whole
    3-line window (which includes the citation's own line, forward text and
    all), so only the short form is one-directional.

    Measured (Issue 753, 2026-09-12) over the 274-row CROSS set: **1** row has
    an owner's alias trailing within 40 chars, and reading it settles the
    question against widening — riir-game-sdk's ``chain_viz` (Plan 032, DeFi
    dashboard). The chain viz is` matches on `chain` in *prose about the
    crate*, not an attribution to riir-chain. So widening forward buys 0
    genuine repairs and SUPPRESSES 1 true finding (itself a crate-hint row,
    the class Issue 751 T2(a) ruled must be counted, not excused).

    That is the backward-only window's argument (Issue 752) on a second axis,
    and it lands the same way for the same reason: an emitted false positive is
    read and dismissed, a suppressed row is invisible to the sample that
    measures the error rate. Lead-only stays — MEASURED, not assumed."""
    m = re.search(rf"\b(?i:{kind})s?\s+0*{n}\b", line)
    if not m:
        return set()
    trail = line[m.end():m.end() + _ALIAS_REACH]
    # The on-disk spellings (Issue 846) are NOT here: a spelling trailing on
    # the citation's own line is already inside the 3-line window, so it
    # QUALIFIES — it is not a suppression cost the way a short alias is.
    return {s.name for s in sibs for a in aliases(s.name)[1:]
            if re.search(rf"\b{re.escape(a)}\b", trail)}


def is_qualified(named: set[str], owners: list[str]) -> bool:
    """A repo name qualifies a citation only if that repo OWNS the number.

    ⛔ The predicate used to be `named != {}` — "is a repo named?", never "does
    that repo own it?". Measured over the workspace (Issue 752): of 368
    qualified citations, **45 named no owner at all** — 37 where the 3-line
    window merely contained a sibling name (a crate-inventory table row, an
    adjacent unrelated clause) and 8 carrying an explicit attribution to a repo
    that does not have the number. `riir-chain Plan 211` where riir-chain's
    `.plans` tops out at 058; `katgpt-rs Issue 513` where 513 is riir-train's.
    Every one of the 45 read as CLEAN.

    ⛔ The census's own `riir-mmorpg-examples Issue 059` example was REFUTED by
    Issue 754 — that repo does own 059, in a heading no file walk could see.
    Owner-consistency is only as sound as `allocated()`, which is why the
    heading path exists and why its filters are measured, not assumed.

    This is Issue 751 T2(a)'s argument, applied to the path it was never
    applied to. Crate hints were counted as findings *because* a plausible
    address that is wrong is worse than no address — and the directory-name
    path, which silently absolves rather than merely annotating, was exempted
    from it with no measurement behind the exemption.

    Owner-consistency applies **exactly when the number has owners**. With no
    owner anywhere in the workspace there is nothing to be consistent with, the
    row is a dangling reference rather than a rebinding hazard, and any named
    repo is accepted — ORPHAN is a different repair and keeps its own bucket.
    """
    return bool(named & set(owners)) if owners else bool(named)


def citations(text: str) -> list[tuple[int, str, int, str]]:
    """(line, kind, number, lead-text) for every citation, list forms expanded."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        for m in _HEAD.finditer(line):
            kind = _CANON[m.group(1).lower()]   # `plan 22` keys as Plan (Issue 924)
            lead = line[max(0, m.start() - _ALIAS_REACH):m.start()]
            out.append((i, kind, int(m.group(3)), lead))
            if not m.group(2):  # singular "Issue 47" never heads a list
                continue
            pos = m.end()
            while (t := _TAIL.match(line, pos)):
                out.append((i, kind, int(t.group(1)), lead))
                pos = t.end()
    return out


def selftest() -> list[str]:
    """Known-answer arms over every rule that decides a citation's bucket.

    Issue 789. This was the largest of the six docs_gate CHECKS running with no
    test of its own arithmetic — and the one with the worst record, because
    AGENTS.md documents it having been WRONG twice in the direction that
    absolves: Issue 752's `named != {}` predicate ("is a repo named?", never
    "does that repo own it?") read 45 rows as clean, and Issue 754 then refuted
    the census that found them, because every read asked the same blind
    `allocated()` the same question.

    So the arms are aimed at the SUPPRESSING paths first: `is_qualified`'s
    owner-consistency and its ORPHAN branch, `heading_allocated`'s two filters
    (the only path here that can make a finding disappear), and the
    lead-only/window split that decides what counts as an address at all. Each
    ⚑ arm reproduces a measured historical defect by name.

    Pure functions plus one temp repo — runs unconditionally at the top of
    `main`, before any pin is read.
    """
    import contextlib
    import io
    import tempfile

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    # ── citations(): a plural kind word may head a LIST ───────────────────
    def cites(text: str):
        return [(k, n) for _ln, k, n, _lead in citations(text)]

    eq("a singular citation", cites("see Issue 749 for the rule"), [("Issue", 749)])
    eq("⚑ a comma list is expanded (Issues 724, 725)",
       cites("a number allocated twice (Issues 724, 725)"),
       [("Issue", 724), ("Issue", 725)])
    eq("⚑ a slash list is expanded (Issues 490/493)",
       cites("orchard drift (Issues 490/493)"),
       [("Issue", 490), ("Issue", 493)])
    eq("an 'and' list is expanded",
       cites("Plans 596 and 597"), [("Plan", 596), ("Plan", 597)])
    eq("a SINGULAR kind word never heads a list",
       cites("Issue 724, 725 rows"), [("Issue", 724)])
    # Every number here is >= 2 digits on purpose: "Issue 1" is outside the
    # width bound and its absence would read as a missing KIND word.
    eq("every kind word is read", sorted({k for k, _ in cites(
        "Issue 11 Plan 22 Research 333 Bench 4444 Proposal 55")}),
       ["Bench", "Issue", "Plan", "Proposal", "Research"])
    eq("a leading zero is not a separate number",
       cites("Issue 059 is closed"), [("Issue", 59)])
    # ⚑ Issue 924: a lowercase house style was invisible to a case-sensitive
    # head — seal-game-editor writes `plan 22` / `issues 724, 725` in prose.
    # The matched word must come back CANONICAL so every bucket keys one spelling.
    eq("⚑ a lowercase citation reads as its canonical kind",
       cites("the plan 22 lane"), [("Plan", 22)])
    eq("⚑ a lowercase plural list is expanded",
       cites("(issues 724, 725)"), [("Issue", 724), ("Issue", 725)])
    # The width bound is load-bearing (Issue 753), not a blind spot to widen.
    eq("a single-digit citation is outside the width bound", cites("Issue 7"), [])
    eq("unseen_by_width counts what the bound cannot see",
       unseen_by_width("Issue 7 and Issues 12, 3"), (1, 1))
    eq("⚑ unseen_by_width counts the lowercase form too (Issue 924)",
       unseen_by_width("issue 7 today"), (1, 0))
    # ⚑ The LEAD window, added by Issue 790 T3. `citations` hands `qualifiers`
    # `line[m.start() - _ALIAS_REACH : m.start()]`, and nothing asserted its
    # extent: an off-by-one flip on that subtraction changes which prose counts
    # as sitting ON a citation, which is exactly what decides an alias
    # qualification. The window is 40 chars and is measured at its edge.
    pad = "x" * (_ALIAS_REACH - len("riir-chain "))
    lead_in = [lead for _ln, _k, _n, lead in citations(pad + "riir-chain Issue 750")]
    eq("a name at the far edge of the lead window is IN it",
       [("riir-chain" in l) for l in lead_in], [True])
    lead_out = [lead for _ln, _k, _n, lead in
                citations("riir-chain " + "x" * _ALIAS_REACH + " Issue 750")]
    eq("a name past the lead window is OUT of it",
       [("riir-chain" in l) for l in lead_out], [False])
    eq("the lead window never runs off the start of the line",
       [lead for _ln, _k, _n, lead in citations("Issue 750")], [""])

    # ── aliases(): a short form is only offered when it is unambiguous ────
    eq("the full name is always a form", aliases("katgpt-rs"), ["katgpt-rs"])
    eq("a 5-char stem earns an alias",
       aliases("riir-chain"), ["riir-chain", "chain"])
    # ⚑ EXACTLY 4 is the documented boundary and nothing sat on it: every arm
    # used a 5-char or a 2/3-char stem, so `arm_reach_audit` reported the
    # `>= 4` surviving a `>= -> >` flip (Issue 790 T3). `riir-auth` is the
    # live repo whose stem is exactly 4.
    eq("a stem of exactly 4 earns an alias",
       aliases("riir-auth"), ["riir-auth", "auth"])
    eq("⚑ a 2-char stem does not ('ai' collides with prose)",
       aliases("riir-ai"), ["riir-ai"])
    eq("a 3-char stem does not", aliases("riir-dao"), ["riir-dao"])
    eq("a non-riir name has no stem", aliases("mmorpg-remake"), ["mmorpg-remake"])
    eq("⚑ the on-disk spelling is a separate accessor (Issue 846)",
       spelling_aliases("mmorpg-remake"), ["seal-remake"])
    eq("an unaliased repo has no spelling",
       spelling_aliases("riir-ai"), [])

    # ── _NAME: a repo name is only a name when nothing extends it ─────────
    eq("⚑ mmorpg-remake-unity does not name mmorpg-remake",
       bool(_NAME["mmorpg-remake"].search("riir-viewbridge names mmorpg-remake-unity")),
       False)
    eq("⚑ riir-games-mmorpg does not name riir-game-sdk",
       bool(_NAME["riir-game-sdk"].search("in riir-games-mmorpg")), False)
    eq("a possessive still names the repo",
       bool(_NAME["riir-ai"].search("riir-ai's scripts")), True)
    eq("a path component still names the repo",
       bool(_NAME["riir-ai"].search("../riir-ai/scripts/x.py")), True)

    # ── Issue 846: the ON-DISK spellings of the aliased repos ─────────
    # The contract names mmorpg-editor / mmorpg-remake / mmorpg-remaster
    # match no directory on either measured box; every document citing their
    # plans was written against the on-disk spelling (44 CROSS rows at the
    # first real read, nearly every specimen naming the owner correctly in
    # prose the matcher could not see). A spelling qualifies in exactly the
    # two places the contract full name does — the lead and the 3-line
    # window — and NEVER accuses: `written_names` stays contract-only, so a
    # spelling can clear a row but never produce a ⛔MISATTRIBUTED.
    spell_sibs = [Path("/w/mmorpg-remake")]

    def spell_quals(lines, ln, lead):
        w, a = qualifiers(lines, ln, lead, spell_sibs)
        return sorted(w), sorted(a)

    eq("⚑ a spelling ON the citation qualifies (the lead)",
       spell_quals(["seal-remake Issue 011"], 1, "seal-remake "),
       (["mmorpg-remake"], ["mmorpg-remake"]))
    eq("⚑ a spelling in the 3-line window qualifies",
       spell_quals(["authored at seal-remake", "", "see Issue 011"], 3, "see "),
       (["mmorpg-remake"], []))
    eq("⚑ a spelling trailing ON the citation's own line qualifies "
       "(the window reads forward text)",
       spell_quals(["Issue 011 landed in seal-remake later"], 1, ""),
       (["mmorpg-remake"], []))
    eq("⚑ a LONGER name does not name the repo (seal-remake-unity)",
       spell_quals(["seal-remake-unity Issue 011"], 1, "seal-remake-unity "),
       ([], []))
    eq("⛔ a spelling NEVER accuses (written_names stays contract-only)",
       sorted(written_names("seal-remake Issue 500", spell_sibs)), [])

    # ── qualifiers(): window vs adjacent, and the alias's one direction ───
    sibs = [Path("/w/riir-ai"), Path("/w/riir-chain"), Path("/w/riir-train")]

    def quals(lines, ln, lead):
        w, a = qualifiers(lines, ln, lead, sibs)
        return sorted(w), sorted(a)

    eq("a full name ON the citation is both window and adjacent",
       quals(["riir-ai Issue 750"], 1, "riir-ai "),
       (["riir-ai"], ["riir-ai"]))
    eq("a full name in the 3-line window is window-only",
       quals(["riir-ai owns this area", "", "see Issue 750"], 3, "see "),
       (["riir-ai"], []))
    eq("the window reaches exactly 3 lines back, not 4",
       quals(["riir-ai owns this", "", "", "see Issue 750"], 4, "see "),
       ([], []))
    eq("⚑ an alias is accepted ON the citation",
       quals(["chain Issue 27"], 1, "chain "),
       (["riir-chain"], ["riir-chain"]))

    # ⛔ written_names(): the ACCUSATION half, and the one measured case.
    # The alias pooling above is a LENIENCY when it qualifies a row and a false
    # ACCUSATION when it attributes one, and `⛔MISATTRIBUTED` is walled at 0.
    # riir-game-sdk's "the Active-preview mirror client chain (Plan 199 ...)"
    # was reported as naming riir-chain; the string `riir-chain` is absent from
    # that whole file.
    eq("written_names accepts a full directory name",
       sorted(written_names("riir-ai Issue 750", sibs)), ["riir-ai"])
    eq("⛔ written_names REFUSES a bare alias — an accusation must be able "
       "to quote the address it says was written",
       sorted(written_names("the mirror client chain ", sibs)), [])
    eq("...while qualifiers still accepts that same alias (leniency is "
       "one-directional)",
       quals(["the mirror client chain Issue 27"], 1,
             "the mirror client chain "), (["riir-chain"], ["riir-chain"]))
    eq("written_names honours the segment boundary too",
       sorted(written_names("in riir-games-mmorpg ", sibs)), [])
    eq("written_names reads every name on the lead, not just the first",
       sorted(written_names("riir-ai and riir-train ", sibs)),
       ["riir-ai", "riir-train"])
    eq("⚑ an alias in the WINDOW qualifies nothing ('train' is ordinary prose)",
       quals(["we train the model", "", "see Issue 750"], 3, "see "),
       ([], []))

    # ── alias_trail_owners(): lead-only, measured not assumed ─────────────
    eq("a trailing alias is REPORTED, not accepted",
       sorted(alias_trail_owners("see Issue 27 in the chain repo", "Issue", 27, sibs)),
       ["riir-chain"])
    eq("no trailing alias is an empty set",
       alias_trail_owners("see Issue 27 for details", "Issue", 27, sibs), set())
    eq("a citation the line does not carry reports nothing",
       alias_trail_owners("see Issue 27 in chain", "Issue", 99, sibs), set())

    # ── is_qualified(): OWNER-consistency, the Issue 752 repair ───────────
    eq("a named owner qualifies", is_qualified({"riir-train"}, ["riir-train"]), True)
    eq("⚑ a named NON-owner does not (the 45-row class)",
       is_qualified({"katgpt-rs"}, ["riir-train"]), False)
    eq("no name at all does not qualify", is_qualified(set(), ["riir-train"]), False)
    # ORPHAN: with no owner anywhere there is nothing to be consistent with.
    eq("an unowned number accepts any named repo",
       is_qualified({"katgpt-rs"}, []), True)
    eq("an unowned number with no name is still unqualified",
       is_qualified(set(), []), False)

    # ── fenced_lines(): FAIL-SAFE on an unterminated fence ────────────────
    inside, open_at = fenced_lines("a\n```\nb\n```\nc")
    eq("fenced body lines are excluded", sorted(inside), [1, 2, 3])
    eq("a terminated file reports no open fence", open_at, None)
    # ⚑ The fixture needs a TERMINATED block BEFORE the unterminated one, or
    # the arm cannot see the fail-safe at all: with no closed block there is
    # nothing in `inside` for the discard to discard, and the arm reads INERT
    # under the very perturbation it is aimed at (measured).
    inside, open_at = fenced_lines("a\n```\nb\n```\nc\n```\nd")
    eq("⚑ an unterminated fence discards the WHOLE exclusion set (fail-safe)",
       inside, set())
    eq("an unterminated fence reports where it opened", open_at, 5)

    # ── parse_pins(): `documents` is a list, everything else an int ───────
    with tempfile.TemporaryDirectory() as td:
        pins_path = Path(td) / "pins.txt"
        pins_path.write_text(
            "# a comment\n"
            "documents = AGENTS.md HISTORY.md\n"
            "min_repos = 16   # trailing comment\n"
            "\n", encoding="utf-8")
        eq("pins parse, comments and blanks dropped",
           parse_pins(pins_path), {"documents": ["AGENTS.md", "HISTORY.md"],
                                   "min_repos": 16})
        # ⚑ The `"=" not in line` half of the filter had no arm: a prose line
        # with no `=` must be DROPPED, not partitioned into a key with an empty
        # value (which `int()` would then raise on, from a line nobody expects
        # to be a pin).
        pins_path.write_text("documents = AGENTS.md\n"
                             "a stray prose line with no equals sign\n"
                             "min_repos = 16\n", encoding="utf-8")
        eq("an equals-less line is dropped, not partitioned",
           parse_pins(pins_path), {"documents": ["AGENTS.md"], "min_repos": 16})

    # ── heading_allocated(): the ONLY path that can suppress a finding ────
    # Its two filters are measured here rather than assumed, per its docstring.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "katgpt-rs"
        repo.mkdir()
        docs = _self_docs()
        body = "\n".join([
            # accepted: the number is IMMEDIATELY followed by a parenthetical
            "## Issue 059 (2026-08-14) — Demonstration-teachable pets",
            # ⚑ Issue 781's style gap: a parenthetical that is not adjacent
            "## Issue 097 resolved — some title (2026-01-01)",
            # ⚑ Issue 781 arm 2's measured NEGATIVE: commentary on a number is
            #    not an allocation of it, and no punctuation rule separates
            #    `043 follow-up (…)` from `097 resolved — … (…)`.
            "## Issue 043 follow-up (2026-01-01)",
            # the foreign-repo filter: this repo is not recording its OWN
            "## Issue 511 (riir-train) — not ours to allocate",
            # the kind filter
            "## Plan 222 (2026-01-01) — a plan, not an issue",
            # the fence filter
            "```",
            "## Issue 888 (2026-01-01) — inside a fenced block",
            "```",
            # ── Issue 823: the DATE-LED house style, six repos wide ─────────
            # accepted: same discriminator, one position over — the number is
            # IMMEDIATELY followed by its title delimiter.
            "## 2026-09-16 — Issue 113: the auto-oracle",
            "## 2026-09-16 — Issue 120, the parallel session's side",
            # ⛔ the discriminator is UNCHANGED, so the date-led form rejects
            #    exactly what the leading form rejects. These two are the whole
            #    soundness argument: if either were read, Issue 823 WOULD be
            #    the widening AGENTS.md calls unsound.
            "## 2026-09-16 — Issue 152 resolved: a title",
            "## 2026-09-16 — Issue 044 follow-up: commentary on a number",
            # the foreign filter still applies, over the WHOLE remainder
            "## 2026-09-16 — Issue 512: a riir-train thing, not ours",
            # a date-led heading of another kind
            "## 2026-09-16 — Plan 223: a plan, not an issue",
            # ── Issue 828: the DELIMITER, the third position the rule was
            #    silently anchored to. 56 records workspace-wide, this repo's
            #    own house style, its own newest closes.
            "## Issue 788 — a dash-delimited title, nothing interstitial",
            # ⛔ the discriminator is UNCHANGED again: `follow-up` is still
            #    rejected, now in the NEW delimiter. If this were read, Issue
            #    828 WOULD be the widening AGENTS.md calls unsound.
            "## Issue 789 follow-up — commentary, not an allocation",
            # ⛔ the ASCII hyphen is a delimiter only when space-separated on
            #    both sides. This is a live riir-ai heading shape where the
            #    hyphen is part of a WORD.
            "## Issue 366-class — the hyphen is inside a word",
        ])
        # ONE pinned document, not all of them: `heading_allocated` unions a
        # SET (so duplicates are invisible) but `heading_style_blind` COUNTS,
        # so writing the same body to every pinned doc multiplies its two
        # numbers by len(docs) and the arm stops being readable.
        eq("there is at least one pinned document to write to", bool(docs), True)
        p = repo / docs[0]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        got = heading_allocated(repo, ".issues", ["katgpt-rs", "riir-train"])
        eq("a self-allocating heading is read", 59 in got, True)
        eq("⚑ `NNN resolved — … (date)` is NOT read (the style gap, Issue 781)",
           97 in got, False)
        eq("⚑ `NNN follow-up (date)` is NOT an allocation (arm 2's negative)",
           43 in got, False)
        eq("⚑ a heading naming a FOREIGN repo does not allocate here",
           511 in got, False)
        eq("a heading of another kind does not allocate", 222 in got, False)
        eq("a heading inside a fence does not allocate", 888 in got, False)
        eq("⚑ Issue 823: a DATE-LED allocation heading is read", 113 in got, True)
        eq("⚑ Issue 823: the date-led comma form is read", 120 in got, True)
        eq("⛔ Issue 823: date-led `NNN resolved:` is NOT read — the "
           "discriminator is unchanged, not dropped", 152 in got, False)
        eq("⛔ Issue 823: date-led `NNN follow-up:` is NOT read — arm 2's "
           "negative survives the new position", 44 in got, False)
        eq("⚑ Issue 823: the foreign filter reads the WHOLE date-led remainder",
           512 in got, False)
        eq("⚑ Issue 823: the kind filter applies to the date-led form",
           223 in got, False)
        eq("⛑ Issue 828: a DASH-delimited allocation heading is read",
           788 in got, True)
        eq("⛔ Issue 828: `NNN follow-up — …` is NOT read — arm 2's negative "
           "survives the new DELIMITER", 789 in got, False)
        eq("⛔ Issue 828: a bare ASCII hyphen inside a word is not a delimiter",
           366 in got, False)
        eq("nothing else was read", sorted(got), [59, 113, 120, 788])
        # The style-blind triage quantity. Issue 823: the meter counts the
        # date-led family too, or it reports a PERFECT score over a house style
        # it cannot see (measured: riir-chain 0/1 over 21 records). Shaped rows
        # surviving the foreign filter are 059, 097, 043 + the four date-led
        # Issue rows (113, 120, 152, 044); 511/512 are foreign, 222/223 another
        # kind, 888 fenced. Accepted: 059, 113, 120.
        eq("heading_style_blind measures the gap, both sides filtered",
           heading_style_blind(repo, ".issues", ["katgpt-rs", "riir-train"]),
           (4, 10))

        # ── Issue 828 T4: the COST meter, which is what ANSWERED Issue 823
        # T5. Armed HERE and not only in the sweep: reach is measured per
        # module, and the predicate that decides it lives in this one.
        # Unread and unknown here: 097, 043, 152, 044, 789, 366 — the accepted
        # rows (059, 113, 120, 788) must never be priced, because the rule
        # change does not touch them.
        names2 = ["katgpt-rs", "riir-train"]
        eq("⛑ every unread record prices when no other oracle knows it",
           heading_unread_novel(repo, ".issues", names2, known=set()), 6)
        eq("⛑ a record another oracle covers cannot change a verdict and is "
           "not priced",
           heading_unread_novel(repo, ".issues", names2, known={97}), 5)
        eq("⛔ every unread number known ⇒ the widening buys NOTHING",
           heading_unread_novel(repo, ".issues", names2,
                                known={97, 43, 152, 44, 789, 366}), 0)
        eq("⛔ knowing an ACCEPTED number changes nothing — it was never in "
           "the residue",
           heading_unread_novel(repo, ".issues", names2,
                                known={59, 113, 120, 788}), 6)

    # ── ci_deferred(): the DEFERRAL, which is this gate's loudest output ──
    # ⚑ Issue 790 T3. Eight of this module's survivors were in here, and it is
    # the path a reader trusts on EVERY partial-clone and CI run — the
    # "instrument alive, adjudication deferred" line. Nothing armed any of it:
    # not the two blindness refusals, not the posture label, not the
    # locally-resolve/cross-repo split that the line reports as fact.
    global REPO_ROOT
    _real_root = REPO_ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            REPO_ROOT = Path(td)
            (REPO_ROOT / ".issues").mkdir()
            for n in ("0123", "0124"):
                (REPO_ROOT / ".issues" / f"{n}_local_thing.md").write_text(
                    "x", encoding="utf-8")
            doc = "D.md"
            # ⚠ TWO local and ONE cross, deliberately ASYMMETRIC: with 1 and 1
            # the `n in local[kind]` test flipped to `not in` produces the
            # identical message, and the arm reads INERT against the very
            # mutation it is aimed at (measured).
            (REPO_ROOT / doc).write_text(
                "see Issue 123 and Issue 124 (ours) and Issue 456 (theirs)\n",
                encoding="utf-8")
            pins = {"max_single_digit": 0, "min_citations_scanned": 3}

            def deferred(posture="CI", **over):
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    rc = ci_deferred({**pins, **over}, [doc], 16, posture=posture)
                return rc, sink.getvalue()

            rc, out = deferred()
            eq("the clean deferral returns 0", rc, 0)
            # The split is REPORTED as fact, so it is pinned as fact: 123 is
            # allocated locally, 456 is not.
            eq("the local/cross split is right",
               "3 citations scanned in 1 document(s); 2 resolve locally, "
               "1 cross-repo" in out, True)
            eq("the deferral says it is NOT an adjudication",
               "this line is not an adjudication" in out, True)
            eq("the CI posture is named", "DOCS_GATE_CI=1" in out, True)
            rc, out = deferred(posture="partial")
            eq("the partial-clone posture is named a DIFFERENT way",
               (rc, PARTIAL_MARKER in out, "DOCS_GATE_CI=1" in out), (0, True, False))

            # Both blindness refusals must exit 2 — a deferral printed over a
            # blind instrument is the one output here that must not exist.
            rc, out = deferred(min_citations_scanned=4)
            eq("a walk below its floor refuses", (rc, "INSTRUMENT" in out), (2, True))
            # Both single-digit FORMS, one at a time. ⚠ `unseen_h + unseen_t`
            # is summed in the condition and again in the message, and with a
            # head-only fixture a `+ -> -` flip is arithmetically identical
            # (1 - 0 == 1 + 0) — so the TAIL form is the one that discriminates
            # it, and both are pinned rather than assumed symmetric.
            for label, body, want_n in (
                ("a single-digit HEAD", "see Issue 123 and Issue 7\n", 1),
                ("a single-digit list TAIL", "see Issues 123, 4\n", 1),
            ):
                (REPO_ROOT / doc).write_text(body, encoding="utf-8")
                rc, out = deferred(min_citations_scanned=1)
                # ⚠ The count is anchored on `INSTRUMENT: ` and not matched as
                # a bare substring: `"-1 single-digit …"` CONTAINS
                # `"1 single-digit …"`, so the loose form passed a flipped sum
                # that printed a negative count (measured).
                eq(f"{label} in scope refuses",
                   (rc, "width bound cannot SEE" in out,
                    f"INSTRUMENT: {want_n} single-digit" in out),
                   (2, True, True))
            # …and a missing pinned document, which is the cheapest way for a
            # deferral to be reported over a file nobody opened.
            (REPO_ROOT / doc).unlink()
            rc, out = deferred(min_citations_scanned=1)
            eq("a missing pinned document refuses",
               (rc, "never opened" in out), (2, True))

            # ── unterminated_fences(): the hazard REPORT, and its address ──
            # It takes the repo as a parameter, so it arms directly. The
            # `open_at + 1` is a 0-to-1-indexed conversion and nothing checked
            # it: a wrong line here points the reader at the wrong fence.
            docs_pinned = _self_docs()
            (REPO_ROOT / docs_pinned[0]).write_text(
                "lead\nprose\n```\nswallowed to EOF\n", encoding="utf-8")
            eq("an unterminated fence is reported at its OPENING line",
               unterminated_fences(REPO_ROOT), [(docs_pinned[0], 3)])
            (REPO_ROOT / docs_pinned[0]).write_text(
                "lead\n```\nclosed\n```\n", encoding="utf-8")
            eq("a terminated document reports nothing",
               unterminated_fences(REPO_ROOT), [])
            # A pinned document that does not exist is SKIPPED, not a crash —
            # the report is a hazard list, and the missing-doc verdict belongs
            # to `ci_deferred`/`main`, not here.
            (REPO_ROOT / docs_pinned[0]).unlink()
            eq("an absent pinned document is skipped",
               unterminated_fences(REPO_ROOT), [])
    finally:
        REPO_ROOT = _real_root

    return fails


def ci_deferred(pins: dict, docs: list, n_repos: int, posture: str = "CI") -> int:
    """The marked-deferred verdict: instrument ALIVE, cross-repo
    adjudication DEFERRED.

    Only the locally-decidable axes run here: the pinned documents exist, the
    `\\d{2,4}` width bound's complement is still empty, and the citation walk
    still clears its floor. The ownership half is impossible without the
    sibling workspace and is NOT guessed at — the verdict says so in its last
    line, which is the one docs_gate.sh forwards on a pass.

    `posture` names WHY the workspace is absent: "CI" (DOCS_GATE_CI=1, a
    single checkout) or "partial" (DOCS_GATE_PARTIAL_CLONE=1, a box with a
    subset of the canonical repos — Issue 765). Same instrument-alive
    verdict; a partial clone additionally CANNOT adjudicate citations naming
    its absent repos, so the full path would manufacture MISATTRIBUTED rows.
    """
    local = {k: allocated(REPO_ROOT, d) for k, d in KINDS.items()}
    scanned = local_hits = 0
    unseen_h = unseen_t = 0
    for doc in docs:
        p = REPO_ROOT / doc
        if not p.is_file():
            print(f"✗ INSTRUMENT: pinned document {doc} is missing — a gate cannot "
                  f"report clean over a file it never opened")
            return 2
        text = p.read_text(encoding="utf-8")
        h, t = unseen_by_width(text)
        unseen_h += h
        unseen_t += t
        for _ln, kind, n, _lead in citations(text):
            scanned += 1
            if n in local[kind]:
                local_hits += 1
    if (unseen_h + unseen_t) > pins["max_single_digit"]:
        print(f"✗ INSTRUMENT: {unseen_h + unseen_t} single-digit citation form(s) "
              f"now in scope, over the pinned {pins['max_single_digit']} — the "
              f"width bound cannot SEE them (same verdict as the workstation run)")
        return 2
    if scanned < pins["min_citations_scanned"]:
        print(f"✗ INSTRUMENT: scanned {scanned} citations < floor "
              f"{pins['min_citations_scanned']} — the citation regex went blind")
        return 2
    print(f"  {posture}: {scanned} citations scanned in {len(docs)} document(s); "
          f"{local_hits} resolve locally, {scanned - local_hits} cross-repo")
    label = ("CI scope (DOCS_GATE_CI=1)" if posture == "CI"
             else f"partial-clone scope ({PARTIAL_MARKER}=1)")
    print(f"✓ {label} — instrument alive over {n_repos} repo(s): "
          f"width bound clean, walk above floor; cross-repo adjudication DEFERRED "
          f"to the full-workspace docs_gate run — this line is not an adjudication")
    return 0


def main() -> int:
    # The canary runs BEFORE the deferral branches, not after. This gate's
    # loudest posture on a partial clone and in CI is an instrument-ALIVE
    # deferral — a line that asserts the classifier works and the adjudication
    # is owed elsewhere. A deferral printed on top of a broken classifier is
    # the one output here that must not be possible, so the arms gate it too.
    arm_failures = selftest()
    if arm_failures:
        print("✗ INSTRUMENT: issue_citation_gate's own selftest does not pass. Every "
              "path below — verdict AND deferral — would be unreadable; this gate's "
              "suppressing rules have been measured wrong twice (Issues 752, 754):")
        for f in arm_failures:
            print(f)
        return 2

    pins = parse_pins(FLOORS)
    docs = pins["documents"]
    assert isinstance(docs, list)

    repos = contract_repos(WORKSPACE)
    sibs = [r for r in repos if r.resolve() != REPO_ROOT]
    # The partial-clone axis (Issue 765): a marked box with a SUBSET of the
    # canonical repos defers the cross-repo half exactly like CI — not only
    # below the floor, because a 15-of-20 box would run the full path and
    # manufacture MISATTRIBUTED rows for citations naming the 5 absent repos
    # (absent repos cannot own anything). The marker is explicit, never
    # auto-detected; a full box (walk == snapshot) ignores it entirely.
    _, absent, _ = partial_clone_state(sorted(r.name for r in repos))
    marker_partial = (os.environ.get(PARTIAL_MARKER) == "1"
                      and bool(absent) and len(repos) > 1)
    if len(repos) < pins["min_repos"] or marker_partial:
        if os.environ.get("DOCS_GATE_CI") == "1":
            return ci_deferred(pins, docs, len(repos))
        if marker_partial:
            return ci_deferred(pins, docs, len(repos), posture="partial")
        if len(repos) > 1:
            print(f"✗ INSTRUMENT: derived {len(repos)} contract repos < floor "
                  f"{pins['min_repos']} — the population went blind; every ceiling "
                  f"below would pass vacuously. Either this box is a PARTIAL "
                  f"CLONE (canonical repos simply not cloned here; the snapshot "
                  f"names {len(absent)} absent: {absent}) — export "
                  f"{PARTIAL_MARKER}=1 for the instrument-alive deferral, do "
                  f"NOT regenerate anything — or it is a full workstation and a "
                  f"sibling moved or went missing: fix the walk, do not mark "
                  f"the box.")
        else:
            print(f"✗ INSTRUMENT: derived {len(repos)} contract repos < floor "
                  f"{pins['min_repos']} — the population went blind; every ceiling "
                  f"below would pass vacuously. A single checkout cannot "
                  f"adjudicate cross-repo citations — the CI lane marks the "
                  f"context with DOCS_GATE_CI=1 for the deferred verdict; a "
                  f"workstation run must fix the walk instead.")
        return 2

    # An unterminated fence disarms the allocation path's fence filter (it fails
    # SAFE, excluding nothing), so the gate would still be correct — but it is a
    # real rendering bug and the filter's premise, so it REDS rather than being
    # noted. Scoped to the repos whose allocations this verdict rests on.
    stray = [(r.name, d, ln) for r in repos for d, ln in unterminated_fences(r)]
    if stray:
        for name, doc, ln in stray:
            print(f"✗ INSTRUMENT: {name}/{doc}:{ln} opens a fenced code block that is "
                  f"never closed — the tail of that file renders as code, and the "
                  f"allocation path's fence filter is disarmed over it")
        return 2

    local = {k: allocated(REPO_ROOT, d) for k, d in KINDS.items()}
    elsewhere = {k: {} for k in KINDS}
    for r in sibs:
        for k, d in KINDS.items():
            for n in allocated(r, d):
                elsewhere[k].setdefault(n, []).append(r.name)

    scanned = 0
    unseen_h = unseen_t = 0
    unqualified: list[str] = []
    ambiguous: set[tuple[str, int]] = set()
    for doc in docs:
        p = REPO_ROOT / doc
        if not p.is_file():
            print(f"✗ INSTRUMENT: pinned document {doc} is missing — a gate cannot "
                  f"report clean over a file it never opened")
            return 2
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        h, t = unseen_by_width(text)
        unseen_h += h
        unseen_t += t
        for ln, kind, n, lead in citations("\n".join(lines)):
            scanned += 1
            owners = elsewhere[kind].get(n, [])
            if n in local[kind]:
                if owners:
                    ambiguous.add((kind, n))
                continue
            # Cross-repo: a sibling repo name within the citation's own
            # paragraph-scale context (3 lines) is what OFFERS an address.
            # THREE lines, and the window size is a MEASURED trade-off, not a
            # guess. This prose hard-wraps at 80 columns, so a single sentence
            # routinely spans 2-3 lines ("Downstream: riir-ai\nIssue 912 T4's"
            # is one attribution split by a line break). Tightening to
            # same-line-only was measured at **4 false positives**, all of that
            # shape. The cost of the wider window is the opposite error: an
            # unrelated repo name that happens to sit within 3 lines is offered
            # as an address for a citation that names nothing — a
            # `riir-train/data/*.gguf` path two lines up in a model list does
            # it. Both directions are real; 3 lines is where the errors were
            # fewest. What makes the OFFER an ANSWER is `is_qualified` —
            # the named repo has to own the number (Issue 752).
            named, adj = qualifiers(lines, ln, lead, sibs)
            if is_qualified(named, owners):
                continue
            where = "/".join(owners) if owners else "NO REPO IN THE WORKSPACE"
            why = "names no repo"
            bad = adj - set(owners)
            if bad:
                why = (f"⛔MISATTRIBUTED — names {'/'.join(sorted(bad))}, "
                       f"which does NOT own {n}")
            elif named:
                why = (f"the only repo in its window is "
                       f"{'/'.join(sorted(named))}, which does NOT own {n}")
            unqualified.append(f"  {doc}:{ln}  {kind} {n} — lives in {where}, "
                               f"{why}\n      {lines[ln - 1].strip()[:120]}")

    if (unseen_h + unseen_t) > pins["max_single_digit"]:
        print(f"✗ INSTRUMENT: {unseen_h + unseen_t} single-digit citation form(s) "
              f"({unseen_h} head, {unseen_t} list-tail) now in scope, over the pinned "
              f"{pins['max_single_digit']} — the `\\d{{2,4}}` width bound cannot SEE them, "
              f"so a clean verdict no longer covers the scope. Adjudicate the rows: they "
              f"are citations (qualify them, and widen the bound with the 51-false-head "
              f"cost re-measured) or section numbering (leave both alone).")
        return 2

    if scanned < pins["min_citations_scanned"]:
        print(f"✗ INSTRUMENT: scanned {scanned} citations < floor {pins['min_citations_scanned']} — "
              f"the citation regex went blind, not the prose clean")
        return 2

    print(f"  scanned {scanned} citations in {len(docs)} document(s) over "
          f"{len(repos)} contract repos ({len(sibs)} siblings)")
    # REPORTED, never gated — see the module docstring. This is the size of
    # what the gate cannot decide, printed next to the verdict so a green is
    # never mistaken for a green over everything.
    print(f"  AMBIGUOUS (local AND sibling — undecidable by number, NOT a pass): {len(ambiguous)}")
    # The width bound's own complement, re-counted rather than remembered: a
    # blind spot recorded as empty ONCE is a claim about a corpus five-plus
    # sessions edit daily (Issue 753).
    print(f"  width bound `\\d{{2,4}}`: {unseen_h + unseen_t} single-digit form(s) "
          f"in scope (pinned max {pins['max_single_digit']}) — NOT scanned, by design")

    if unqualified:
        print(f"✗ issue citation gate FAILED — {len(unqualified)} unqualified cross-repo citation(s)")
        for row in unqualified:
            print(row)
        print("  Fix: name the owning repo in the prose — `riir-ai Issue 750`, "
              "`riir-train Issue 513`. The number alone is not an address.")
        if absent:
            # Reached only on the FULL path (a marked partial clone deferred
            # above) — but the walk can still be short of the canonical set on
            # an UNMARKED box above the floor. Those rows are then suspect,
            # and the box, not the prose, is the likely cause.
            print(f"  note: {len(absent)} canonical repo(s) absent on this box "
                  f"({', '.join(absent)}) — citations naming them CANNOT be "
                  f"adjudicated here; if this is a partial clone, export "
                  f"{PARTIAL_MARKER}=1 and re-run for the deferral verdict")
        return 1

    print(f"✓ issue citation gate PASSED — every cross-repo citation names its repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
