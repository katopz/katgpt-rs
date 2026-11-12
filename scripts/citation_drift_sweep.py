#!/usr/bin/env python3
"""Run the citation verdict over EVERY contract repo, not just this one.

`scripts/issue_citation_gate.py` (Issue 749) is katgpt-rs-scoped by
construction: `docs_gate.yml` has a single checkout, so it could never see a
sibling. That is the right shape for a per-push CI gate and the wrong shape
for this defect class, which is *especially* not repo-local — a citation's
whole problem is that its referent lives somewhere else. This is the sixth
instance of the shape (Issue 702 `ci_gate_coverage`, 725 `numbering`,
`required_features`, `percentile`, `trap_sentinel`), and like the first two it
found real rows the moment it was pointed anywhere but here.

    this script                 workstation, on demand, every contract repo
    issue_citation_gate.py      CI, per-push (docs_gate.sh), katgpt-rs only

⛔ REPORT, exit 0-on-clean / 1-on-drift / 2-on-untrustworthy — NOT a per-push
gate, for the same reason as every other sweep in the family: CI's single
checkout would derive an EMPTY population and print a confident green over
zero repos.

Measured (Issue 752, 2026-09-12), over 19 contract repos / 2,948 citations
in 35 documents — a dated SNAPSHOT, not a checksum: five-plus concurrent
sessions edit these documents, and the walk moved by 3 between this repair's
first run and its last:

    291 CROSS over 164 per-repo adjudications · 54 IN-LOCAL-RANGE ·
    0 ORPHAN · 908 AMBIGUOUS
    (of the 291: 7 ⛔MISATTRIBUTED, 22 ⛔MISLEADING crate hints)

**291 counts EDITS; 164 counts DECISIONS, and sizing the work from the first
is wrong by up to 4.2x.** Inserting a repo name is mechanical; deciding WHICH
owner a sentence means — most of these numbers are owned by several repos at
once — is paid once per number, not once per occurrence. riir-viewbridge's 17
rows are FOUR decisions (`Plan 532` alone recurs nine times); riir-auth's 8
are three; riir-chain's 4 are four. Reported per repo as `cross=N over M num`,
with the same standing as tail support in the percentile audit: it ORDERS the
work and is never a second verdict.

The workspace total is a SUM of the per-repo counts, never a union — riir-auth
and riir-game-sdk both citing `Plan 488` is two adjudications in two
documents, and unioning them reported 142 where the work is 164.

⛔ **TWO error rates over TWO populations — they are never blended**, because
a SAMPLE rate does not transfer to rows it never sampled:

    254 rows (the pre-752 corpus)   7/43 = 16%, a stratified SAMPLE read
                                    line-by-line across 13 repos (751 T1)
    +45 rows (owner-consistency)    1/45, a full CENSUS — every row read (752),
                                    with ONE row refuted afterwards by a
                                    measurement the census could not make (754)

The predecessor figure — 308 — was quoted with no error rate at all and was
contaminated by an alias class worth ~50% of its first four rows (749's
addendum). A classifier's bucket boundaries ARE the finding, so the FP classes
are written down here rather than silently healed:

    alias/abbreviation NOT in the alias table   ("ndb" = riir-neuron-db,
                                                 "mmorpg" = riir-mmorpg-examples)
    alias TRAILING the citation                 (the alias reach is a 40-char
                                                 LEAD; the full directory name
                                                 is read from the whole window,
                                                 forward text included. MEASURED
                                                 at 1 of 274 CROSS rows, and that
                                                 one is `chain` inside prose
                                                 about the `chain_viz` crate —
                                                 widening forward buys 0 repairs
                                                 and SUPPRESSES a true finding,
                                                 Issue 753)
    attribution just OUTSIDE the 3-line window  (4 lines up, or 2 lines DOWN —
                                                 the window is backward-only;
                                                 MEASURED and deliberately kept,
                                                 see below)
    intra-document back-reference               ("see Issue 092 above", where
                                                 THIS doc carries a heading for
                                                 the foreign number)
    a local number one above the local max      (riir-ai cites its own Plan 590
                                                 with `.plans` topping out at 588)

and the class that ran the OTHER way — 45 rows the rule ABSORBED
--------------------------------------------------------------
Every FP class above inflates the count. Issue 752 measured the deflating one,
and nobody had looked: qualification asked **"is a repo named?"** and never
**"does that repo own the number?"**. Of 368 citations the rule certified, **45
named no owner at all** — 37 where the 3-line window merely contained a sibling
name (a crate-inventory table row, an adjacent clause) and 8 carrying an
explicit attribution to a repo that does not have the number. `riir-chain Plan
211` where riir-chain's `.plans` top out at 058; `katgpt-rs Issue 513` where
513 is riir-train's and katgpt-rs owns only the script the line is about — the
attribution followed the CODE while the number followed the DOCUMENT.

⛔ The census's THIRD example was WRONG, and it is kept here because the way it
was wrong is the lesson. `riir-mmorpg-examples Issue 059` was filed as an
outright wrong address "where that repo allocated 058 and 061 and never 059".
That repo DID allocate 059 — its own HISTORY.md carries `## Issue 059
(2026-08-14) — Demonstration-teachable pets`, resolved and removed the day it
was filed, and removed WITHOUT an intervening commit, so neither the worktree
walk nor `git log` could see it (Issue 754, `heading_allocated()`). The census
read every row and still could not have caught this: the instrument it checked
each row against was itself blind, so a full census inherits its ORACLE's blind
spots at 100%. `0/45` was a statement about the reader, never about the rule.

That is a **~15% under-count**, the same magnitude as the 16% over-count and
in the opposite direction. It is the T2(a) argument below applied to the path
it was never applied to: crate hints were counted as findings *because* a
plausible address that is wrong beats no address, and the directory-name path
— which silently absolves where the crate path merely annotates — was exempted
from that argument with no measurement behind the exemption.

The repair also tightened the name match to segment boundaries: a plain
`"mmorpg-remake" in ctx` read riir-viewbridge's `mmorpg-remake-unity` as naming
**mmorpg-remake** and qualified a `Plan 031` citation on a different repo's name.

Two adjudications the per-repo filings forced, recorded so they stay decided
---------------------------------------------------------------------------
**`ndb` is NOT a missing alias — it is a real finding.** riir-auth writes "ndb
Plan 327/328" and riir-neuron-db owns both, so the row looks like a false
positive of the alias table. It is not, and the alias table must NOT grow a
hand-typed entry for it. The alias is DERIVED (directory name minus `riir-`),
deliberately, because a hand-maintained list drifts exactly as a hand-typed
repo set does; and more decisively, AGENTS.md is *the contract a new agent
reads*, and a new agent does not know that `ndb` means riir-neuron-db. An
address only its author can follow is the thing this sweep exists to find.

**A truncated read produced a wrong filed count.** riir-viewbridge's issue was
filed at 12 rows when the true count was 13: the per-repo print caps CROSS at
12, and the issue's author counted the printed ROWS instead of reading
`cross=N` in the header. `--full` exists because of this, and the truncation
notice now names the flag — but the durable lesson is that the header is the
count and the row list is a sample of it.

Why the window stays BACKWARD-ONLY — measured, not assumed
----------------------------------------------------------
Owner-consistency made the forward-window question answerable for the first
time, because it strips the noise: ask not "is a repo named below?" but "is an
OWNER of this number named below?". Measured over the CROSS set, **15 rows**
qualify within 1-2 lines forward. Read line by line, **at most 3 are genuine
attributions** — riir-game-sdk's `Issue 458 … See riir-ai/AGENTS.md` is the
clearest. The other twelve are incidental: a dep-path note two lines down
(`../katgpt-rs/crates/katgpt-core`), a `Batch 47 …` list, a "three commits
across repos" enumeration where the named repo is one commit host and not the
plan's owner, and — twice — a forward line attributing a **different** number
(`See .issues/529 (riir-ai)` sitting under a citation of 496 and 528).

So widening forward buys ~3 false-positive repairs and costs ~12 newly
SUPPRESSED true findings. That is the wrong direction by this module's own
governing lesson: a suppressed row is invisible to the sample that produced
the 16% figure, while an emitted false positive is merely read and dismissed.
Backward-only stays.

Three buckets, and the split is the whole point of re-measuring
---------------------------------------------------------------
**CROSS** — the finding. Not local, no repo named, and a sibling DOES own the
number: this is the rebinding hazard (749's founding example).

**IN-LOCAL-RANGE** — UNDECIDED, reported separately, never folded into CROSS.
The number is at or below the repo's OWN top allocation for that kind, so a
locally-intended referent is plausible even though no file carries it: a
number the repo skipped, or filed and never committed. riir-chain's HISTORY
narrating its own work as "Closed as a class (Issue 039, same day)" is
exactly this — riir-chain allocated 1-38 and 41-46, never 39, and six
siblings have one. Calling that a cross-repo citation is a misattribution,
and it was 46 of the 300 rows the first pass reported.

**ORPHAN** — not local, not in local range, and NO repo in the workspace owns
it. Measured 0; kept as its own bucket because it is a different repair
(the number is wrong, or its repo left the contract) from CROSS.

**AMBIGUOUS** is the gate's stated blind spot, carried over verbatim: a number
that exists BOTH locally and in a sibling is undecidable from the number
alone, so it is REPORTED and deliberately NOT gated (a ceiling would red
whenever somebody writes a perfectly correct citation to a new local number a
sibling also happens to have — `staged_set_audit.py`'s rationale one axis over).

The two-lane GAP (Issue 921 Arm B) — preventive, membership-shaped
------------------------------------------------------------------
A repo whose kind dir carries `.highwater_local` (riir-infer `.issues` 035,
riir-rethink 23) DECLARES — machine-readably, per its AGENTS.md numbering
section — that the local lane tops at that counter and the range above it is
inherited-only. For such a repo the GAP is `counter < n < floor`, where floor
is the smallest git-log FILE ADD above the counter (`--diff-filter=A`, full
history — the same record source `file_and_history_allocated` reads; NOT the
worktree walk, and NOT the heading oracle, whose additions are
order-dependent and must never shrink the gap). Citations inside the gap are
NOT local-intent: they classify CROSS/ORPHAN — actionable, with the
mechanical repeat repair — instead of collecting forever as UNDECIDED
IN-LOCAL-RANGE behind a breached ceiling (the measured founding row:
riir-infer's bare `Issue 980`, which sat at n=980 <= top 1004 with no local
record anywhere). Repos without the file derive an empty gap and
byte-identical behavior. The derived floor prints on the per-repo line;
`.highwater_local` rides both advisory patterns (a dirty or upstream-moved
counter can move verdicts, so it is IN this sweep's population and
discloses). `--prove-fires` replays the founding specimen two-sided at
riir-infer `f0191e3^` (the bare 980 must classify CROSS -> riir-ai, gap
35..998) vs `f0191e3` (the repaired row must find nothing): a local CLONE,
not an archive extraction, because the floor leg is a git-log leg and an
extraction has no history — it would derive the tree's smallest add (1003)
and under-test the very derivation the arm exists to pin.

Two audited sub-questions (Issue 751 T2) — both decided with evidence
--------------------------------------------------------------------
**A crate name is NOT a qualification form.** The workspace's crate->repo map
is unique (156 package names, ZERO collisions across 19 repos), so
`riir-games-mmorpg::…` *could* address riir-ai mechanically — but the map
lives in no document the reader has, and the measurement refutes the weaker
claim too: of the 45 CROSS rows carrying a crate name in the window, **4
resolve to a repo that does NOT own the cited number** (riir-game-sdk's
`Issue 097` sits next to `riir-games-mmorpg::sync_facades::avatar_sync` while
097 belongs to riir-mmorpg-examples/riir-chain/mmorpg-editor). Accepting
crate names as qualifiers would certify those four as clean, and they are the
worst rows in the corpus — a plausible address that is wrong. So the rows are
counted as findings and sub-labelled `crate-hint` / `⛔MISLEADING` to ORDER
the repair, never to excuse it.

**Zero-padding is the SAME allocator namespace.** `Issue 006` and `Issue 6`
are one number: both sides already `int()` (the citation regex `\\d{2,4}`, and
`allocated()`'s `^(\\d+)_`). Measured: 5,839 allocated file names, 5,758
3-digit + 81 2-digit, and the 78 numbers carrying BOTH widths are
width-normalisation RENAMES of one document (`.research/07_Screening_…` ->
`.research/007_Screening_…`, same title). Corpus usage is genuinely mixed —
**1,713 padded vs 1,221 unpadded citations** — so treating the forms as
distinct namespaces would misread 58% of it. The width bound `\\d{2,4}` is
**load-bearing, not a blind spot awaiting repair** (Issue 753): re-measured over
every tracked `.md` in the workspace, widening to `\\d{1,4}` would manufacture
**51 false heads** — `## Bench 1: Throughput` is a section NUMBER, not a
citation of `.benchmarks/001_*` — and **0** true ones. The list expander's
PLURAL precondition is the other half and neither may be costed alone: it is
what keeps `Plan 460, 31.5%` and `Issue 096, 2,294 LOC` from ever being read as
a second citation. What the class needed was liveness, not width — the
complement is counted every run and pinned at 0 (`max_single_digit`), because
"0 occurrences" was a dated measurement over documents five-plus sessions edit
daily.

Two floors, not one
-------------------
`max_cross` is green over whatever the instrument can SEE. A regex regression
takes the citation walk to 0 and every ceiling passes, indistinguishable from
clean prose — `min_citations` catches that per repo. It cannot do the job
alone: the finding classes are all decided against the DERIVED repo
population, and a population that collapses to 1 makes every citation read as
local-or-orphan. `min_repos` is that second floor, and it is the same
quantity `issue_citation_floors.txt` owns, so it is ASSERTED equal, not
trusted (the `trap_sentinel_drift_sweep` vs `trap_sentinel_gate` precedent).
`documents` is NOT restated at all — it is read straight out of the gate's
pins, because a quantity with one home cannot drift.

⛔ The corpus MOVES under the sweep. Five-plus agent sessions edit these
documents concurrently; riir-mmorpg-examples' HISTORY.md gained 46 lines
between two runs an hour apart during T1. Ceilings are therefore a RATCHET at
the measured value (numbering_drift_sweep's stance, not trap_sentinel's wall):
a new unqualified citation reds immediately, and the standing backlog is
visible in the pins rather than silently tolerated. Re-pin DELIBERATELY, in
the commit that changes it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the citation regex, the alias table, the allocation walk and the
# document list are the GATE's, so the sweep and the per-push gate can never
# disagree about what a citation IS.
import issue_citation_gate as icg  # noqa: E402
from sweep_population import population_verdict, pin_row_exempt  # noqa: E402

# Issue 842: the derived handles are CONTRACT-named; reads resolve to disk.
import repo_alias  # noqa: E402
from worktree_state import (  # noqa: E402
    STALE_FETCH_HOURS, behind_origin, deferral_line, dirty_files,
    fetch_age_hours, head_text, upstream_axis, worktree_advisory)

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "citation_drift_floors.txt"
GATE_PINS = HERE / "issue_citation_floors.txt"
GATE = HERE / "issue_citation_gate.py"

FIELDS = ("min_citations", "max_cross", "max_in_local_range", "max_orphan")

CROSS, IN_RANGE, ORPHAN = "CROSS", "IN-LOCAL-RANGE", "ORPHAN"
# Issue 794. A SUBSET of IN_RANGE, kept in that bucket for the `max_in_local_range`
# ceiling (the undecided population did not change) and listed separately so the
# 4-row display truncation cannot hide a finding behind undecided noise.
MISATTR_IN_RANGE = "MISATTRIBUTED-IN-RANGE"
# The finding classes, as a tuple, for the Issue-796 worktree-vs-HEAD diff.
# MISATTR_IN_RANGE is deliberately absent: it is a SUBSET of IN_RANGE, and
# including it would count its rows twice on both sides of the comparison.
_CLASSES = (CROSS, IN_RANGE, ORPHAN)

# A crate token is only usable as a REPO HINT when it cannot collide with
# ordinary prose: hyphenated and >= 6 characters. `xtask`, `core`, `cli` are
# package names too and would match every paragraph in the workspace.
_CRATE_MIN = 6
_PKG_NAME = re.compile(r'(?m)^\s*name\s*=\s*"([^"]+)"')


def parse_pins(path: Path) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """`key = value` globals plus one 5-field row per repo. Arity ENFORCED."""
    glob: dict[str, int] = {}
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            glob[key.strip()] = int(val.strip())
            continue
        parts = line.split()
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(
                f"malformed pin row (want {1 + len(FIELDS)} fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(v) for v in parts[1:])))
    return glob, rows


def crate_map(repos: list[Path]) -> dict[str, str]:
    """Every workspace package name -> its owning repo. Derived, never typed.

    Measured ZERO collisions over 156 names / 19 repos, which is what makes a
    crate token a mechanically-unique repo address — and precisely why the
    MISLEADING sub-class is a real finding rather than a parse artefact.
    """
    out: dict[str, str] = {}
    for r in repos:
        # Issue 842: git reads the on-disk directory.
        rs = repo_alias.real(r)
        ls = subprocess.run(["git", "-C", str(rs), "ls-files", "*Cargo.toml"],
                            capture_output=True, encoding="utf-8", errors="replace")
        for rel in ls.stdout.split():
            try:
                text = (rs / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            m = _PKG_NAME.search(text)
            if m and "-" in m.group(1) and len(m.group(1)) >= _CRATE_MIN:
                out.setdefault(m.group(1), r.name)
    return out


def _crate_hits(ctx: str, crates: dict[str, str], patterns: dict[str, re.Pattern]) -> set[str]:
    """Repos named by a CRATE inside the window. Prose writes both
    `riir-games-mmorpg` and `riir_games_mmorpg`; accept either spelling."""
    return {crates[c] for c, rx in patterns.items() if rx.search(ctx)}


def top_allocated(repo: Path, alloc: dict[str, set[int]]) -> dict[str, int]:
    """This repo's OWN ceiling per kind: max(allocated, .highwater).

    `.highwater` is read as well as the files because the Numbering Discipline
    bumps it BEFORE the file lands, and a number in flight is exactly the
    local-intent case IN-LOCAL-RANGE exists to keep out of the finding bucket.
    """
    out = {}
    for kind, sub in icg.KINDS.items():
        top = max(alloc[kind]) if alloc[kind] else 0
        hw = repo / sub / ".highwater"
        if hw.is_file():
            try:
                top = max(top, int(hw.read_text(encoding="utf-8").strip().split()[-1]))
            except (ValueError, IndexError, OSError):
                pass  # numbering_gate.py owns the malformed-.highwater verdict
        out[kind] = top
    return out


FULL = "--full" in sys.argv

# Issue 921 Arm B. The file whose PRESENCE declares the two-lane discipline
# (the repo's AGENTS.md numbering section is the prose half): the local lane
# tops here, the range above is inherited-only. Membership is read from the
# FILE, so a repo without it never enters the gap branch.
HW_LOCAL = ".issues/.highwater_local"


def _local_counter(disk: Path, subdir: str) -> int | None:
    """The declared local-lane top, or None when this kind dir has no lane.

    Malformed content reads as None (no lane) rather than guessing: the
    numbering gates own the counter's format, and an unparseable declaration
    must not invent a gap boundary.
    """
    p = disk / subdir / ".highwater_local"
    if not p.is_file():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip().split()[-1])
    except (ValueError, IndexError, OSError):
        return None


def _file_adds(disk: Path, subdir: str) -> set[int]:
    """Numbers FILE-BACKED in this kind dir across FULL history: git-log
    ADDITIONS only (`--diff-filter=A`), never the worktree walk.

    The same record source `file_and_history_allocated` reads — minus its
    worktree leg, on purpose: a worktree file can be another session's
    in-flight scratch, while an add is a durable allocation event. riir-infer
    998 is the working case: added, later removed, absent from every tree —
    and still the smallest file-backed allocation above that repo's counter.
    """
    log = subprocess.run(
        ["git", "-C", str(disk), "log", "--all", "--diff-filter=A",
         "--name-only", "--pretty=format:", "--", f"{subdir}/"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    out: set[int] = set()
    prefix = re.compile(re.escape(subdir) + r"/(\d+)_")
    for line in log.stdout.splitlines():
        m = prefix.match(line.strip())
        if m:
            out.add(int(m.group(1)))
    return out


def gap_bounds(disk: Path) -> dict[str, tuple[int, int | None]]:
    """The two-lane GAP per kind dir: `{kind: (counter, floor)}` (Issue 921
    Arm B). Empty for every repo without a `.highwater_local` — the branch is
    membership-shaped and those repos are byte-identical.

    floor is the smallest file-backed allocation ABOVE the counter; None when
    no add sits above it, which leaves the gap open — the repo declared its
    local lane tops at the counter, so nothing above can be local-intent.
    Heading-oracle allocations are deliberately EXCLUDED from the floor: an
    Arm-A-style heading addition must not shrink the gap (order-dependence).
    """
    out: dict[str, tuple[int, int | None]] = {}
    for kind, sub in icg.KINDS.items():
        counter = _local_counter(disk, sub)
        if counter is None:
            continue          # no lane declared -> no git-log leg either
        above = sorted(n for n in _file_adds(disk, sub) if n > counter)
        out[kind] = (counter, above[0] if above else None)
    return out


_ROW_KEY = re.compile(r"^(\S+?):(\d+)\s+(\S+)\s+(\d+)\s")


def _row_key(row: str) -> tuple[str, str, str]:
    """A finding row's identity ACROSS two reads of the same document.

    `(document, kind, number)` — deliberately NOT the line number. Comparing
    a worktree row with a HEAD row is the whole point, and any edit above a
    citation shifts its line, so a line-bearing key would report every row in
    an edited document as both UNCOMMITTED and MASKED at once (Issue 797).
    """
    m = _ROW_KEY.match(row)
    return (m.group(1), m.group(3), m.group(4)) if m else ("", "", row[:80])


def _worktree_read(p: Path) -> str | None:
    """The default document source: the WORKING TREE, or None if absent."""
    if not p.is_file():
        return None         # not every repo carries a HISTORY.md — absence
    return p.read_text(encoding="utf-8", errors="replace")


ORACLE_STALE = "ORACLE-STALE"

# The globs naming every directory a number can be allocated in. An oracle
# repo is unreliable for THIS question only if its upstream moved inside one
# of these — a sibling that is behind on source code still answers
# "do you own Plan 226" correctly.
NUMBERED_GLOBS = tuple(f"{d}/*" for d in icg.KINDS.values())


def unreliable_oracles(sibs: list[Path]) -> dict[str, str]:
    """Sibling repos whose "I do not own that number" cannot be believed.

    Issue 827. `is_qualified()` asks a sibling's checkout whether it owns a
    number, and a checkout answers for the commits it has. Measured: a citation
    reading `seal-game-editor Plan 226` — correct, and the qualified form this
    family prescribes — was reported ⛔MISATTRIBUTED and counted CROSS in
    **riir-shader**, because seal-game-editor's checkout sat 260 commits behind
    origin and the allocation landed in that gap. The repo being blamed was
    clean, current, and not the repo whose state produced the verdict.

    Two ways an oracle stops being credible, and they are NOT the same fact:

    * it is BEHIND its upstream on commits touching a numbered directory — the
      allocation may be sitting in those commits;
    * it reports up to date, but from a remote-tracking ref nobody has
      refreshed (Issue 827 T5). `(0, 0)` means "as of the last fetch", and
      before the fetch that exposed it seal-game-editor read exactly `0 behind`
      while hiding those 260 commits.

    ⛔ No network call. Fetching to make our own verdict true would mutate
    another session's refs, which on a box running five concurrent sessions is
    a race — T2's standing refusal.

    Returns `{repo name: why}`, empty when every oracle is current. A repo with
    no upstream is NOT listed: `behind_origin` answers `None` there and the
    repo claims nothing, so its allocation set is simply what it has.
    """
    out: dict[str, str] = {}
    for s in sibs:
        # Issue 842: git reads the on-disk directory; the printed key is the
        # CONTRACT spelling.
        beh = behind_origin(repo_alias.real(s), NUMBERED_GLOBS)
        if beh is None:
            continue
        if beh[1]:
            out[repo_alias.display(s.name)] = (
                f"{beh[0]} commits behind upstream, {beh[1]} of "
                f"them touching a numbered directory")
            continue
        if beh == (0, 0):
            age = fetch_age_hours(repo_alias.real(s))
            if age is None:
                out[repo_alias.display(s.name)] = (
                    "reports up to date, but has never been fetched")
            elif age > STALE_FETCH_HOURS:
                out[repo_alias.display(s.name)] = (
                    f"reports up to date from a remote-tracking ref "
                    f"last refreshed {age:.0f}h ago")
    return out


def audit(repo: Path, sibs: list[Path], alloc: dict[str, dict[str, set[int]]],
          docs: list[str], crates: dict[str, str],
          patterns: dict[str, re.Pattern], read=_worktree_read,
          unreliable: dict[str, str] | None = None) -> dict:
    """One repo -> the three finding classes + BOTH populations under them.

    `read(path) -> str | None` is injected (Issue 797) so the SAME classifier
    can be pointed at HEAD's blobs instead of the working tree. `None` means
    "this document does not exist in the source being read", and it is not a
    finding either way — but it must stay distinguishable from empty text, or
    a repo with no HISTORY.md and a repo whose HISTORY.md is empty read alike.
    """
    mine = alloc[repo.name]
    # Issue 842: the reads go to the ON-DISK directory; every NAME (alloc
    # keys, qualifier matching) stays the CONTRACT spelling of the handle.
    disk = repo_alias.real(repo)
    top = top_allocated(disk, mine)
    gaps = gap_bounds(disk)   # Issue 921 Arm B — empty without .highwater_local
    elsewhere: dict[str, dict[int, list[str]]] = {k: {} for k in icg.KINDS}
    for s in sibs:
        for kind in icg.KINDS:
            for n in alloc[s.name][kind]:
                elsewhere[kind].setdefault(n, []).append(s.name)

    got = {"n_docs": 0, "n_cites": 0, "ambiguous": set(), "misleading": 0,
           "misattributed": 0, MISATTR_IN_RANGE: [],
           "cross_units": set(), "repeat": 0,
           "unseen_width": 0, "alias_trailing": 0, "gap": gaps,
           CROSS: [], IN_RANGE: [], ORPHAN: [], ORACLE_STALE: []}
    for doc in docs:
        text = read(disk / doc)
        if text is None:
            continue          # absence is not a finding, but the doc COUNT is
        got["n_docs"] += 1    # printed, so it must not silently include it
        lines = text.splitlines()
        # The width bound's complement, re-counted every run rather than
        # remembered from one dated measurement (Issue 753).
        got["unseen_width"] += sum(icg.unseen_by_width(text))
        # Two passes over the same document: the first records which numbers
        # the prose DOES attribute somewhere, so the second can tell a bare
        # citation that is unfollowable from one whose attribution is already
        # in this very file a few lines away.
        qualified_here: set[tuple[str, int]] = set()
        for ln, kind, n, lead in icg.citations("\n".join(lines)):
            if n in mine[kind]:
                continue
            owners = elsewhere[kind].get(n, [])
            nmd, _ = icg.qualifiers(lines, ln, lead, sibs)
            if icg.is_qualified(nmd, owners):
                qualified_here.add((kind, n))
        for ln, kind, n, lead in icg.citations("\n".join(lines)):
            got["n_cites"] += 1
            owners = elsewhere[kind].get(n, [])
            if n in mine[kind]:
                if owners:
                    got["ambiguous"].add((kind, n))
                continue
            # ── the gate's qualification rule, REUSED not re-implemented:
            # 3-line window for the full directory name, alias only ON the
            # citation, and the named repo must OWN the number (Issue 752).
            named, adj = icg.qualifiers(lines, ln, lead, sibs)
            if icg.is_qualified(named, owners):
                continue
            # ── Issue 827: the ORACLE's freshness is part of the verdict ────
            # Qualification just failed, and it failed by asking sibling
            # checkouts who owns this number. If the author named a repo whose
            # answer this box cannot believe, the failure is not evidence about
            # the citation — it is evidence about our checkout of that repo.
            # UNDECIDED: never clean (the citation may really be wrong), never
            # counted (a ceiling breached by another repo's fetch schedule is
            # the cries-wolf failure this family refuses).
            #
            # ⛔ It must sit HERE and not on the ⛔MISATTRIBUTED tag below.
            # The measured case was counted as CROSS, and the tag is only its
            # sub-label — suppressing the label alone would have left the
            # ceiling breached and the red standing.
            stale_named = sorted(set(named) & set(unreliable or {}))
            if stale_named:
                why = "; ".join(f"{nm}: {(unreliable or {})[nm]}"
                                for nm in stale_named)
                got[ORACLE_STALE].append(
                    f"{doc}:{ln}  {kind} {n} names {'/'.join(stale_named)} "
                    f"[⚠ ORACLE-STALE — {why}. This box cannot say whether "
                    f"that repo owns the number, so this row is UNDECIDED: "
                    f"not clean, not counted. `git fetch` there and re-run]")
                continue
            ctx = "\n".join(lines[max(0, ln - 3):ln])
            hint = _crate_hits(ctx, crates, patterns) - {repo.name}
            # Issue 921 Arm B: the two-lane gap. A repo carrying
            # `.highwater_local` declares its local lane tops at the counter,
            # so counter < n < floor is NOT local-intent and must not park in
            # UNDECIDED IN-LOCAL-RANGE — it is an actionable CROSS/ORPHAN,
            # with the mechanical repeat repair. Membership-shaped: `gaps` is
            # empty without the file and this branch is dead for that repo.
            g = gaps.get(kind)
            in_gap = g is not None and g[0] < n and (g[1] is None or n < g[1])
            cls = ((CROSS if owners else ORPHAN) if in_gap
                   else (IN_RANGE if n <= top[kind]
                         else CROSS if owners else ORPHAN))
            tag = ""
            gap_note = (
                f"  [two-lane gap {g[0]}..{g[1] if g[1] is not None else 'open'}: "
                f"the local lane tops at {g[0]}, the range above is inherited]"
            ) if in_gap else ""
            # ⛔ `written_names`, NOT `adj`. The accusation half must be able
            # to QUOTE the address it says was written, and `adj` pools full
            # directory names with short-form ALIASES. An alias match that
            # qualifies a citation is a leniency; one that attributes it is a
            # false accusation in a class walled at 0. Measured 2026-09-15:
            # riir-game-sdk's "the Active-preview mirror client chain (Plan 199
            # …)" was reported as naming riir-chain, a string absent from that
            # file, and hard-failed that repo's sweep.
            bad = icg.written_names(lead, sibs) - set(owners)
            if cls is IN_RANGE and bad:
                # Issue 794. IN-LOCAL-RANGE is "UNDECIDED, never clean"
                # because a local referent that was skipped or never committed
                # is plausible. This row REFUTES that premise with its own
                # text: it reached this bucket only because `n not in mine`
                # (the Issue-754 oracle — worktree AND git log AND headings —
                # found no local allocation), and the author wrote a DIFFERENT
                # repo's name directly ON the citation. Followable, to the
                # wrong place. Its own class, never pooled into CROSS: CROSS
                # is unfollowable, and the two repairs read differently.
                #
                # ⛔ The boundary is MEASURED and it is NOT the obvious one.
                # The same predicate at the `n in mine` short-circuit one
                # branch up is 19 rows workspace-wide and 19 of them are
                # FALSE — the prose contrasting a local number with a remote
                # one, the 40-char lead catching the NEIGHBOUR's address
                # (`riir-ai Issue 853 / this repo's Issue 093`, ``in
                # `riir-neuron-db/src/local_kv.rs` (Issue 043``). That is the
                # mechanism, not luck: a locally-allocated number HAS a local
                # referent for the prose to contrast against. So the rule
                # stops here, and the exemption is a measurement rather than
                # an oversight. The IN-RANGE column is n = 1 — "0 false" there
                # is one row's worth of evidence, not a rate.
                got[MISATTR_IN_RANGE].append(f"{doc}:{ln}  {kind} {n}")
                tag = (f"  [⛔MISATTRIBUTED-IN-RANGE: names "
                       f"{'/'.join(sorted(bad))}, which does NOT own {n}; "
                       f"the local-range excuse does not apply — {repo.name} "
                       f"never allocated {n}]")
            elif cls is CROSS and bad:
                # An explicit attribution sitting ON the citation that names a
                # repo without the number. Same standing as the crate hint: it
                # ORDERS the repair, it is not a verdict. Hand-adjudicated at
                # landing, 3 of 8 were genuinely wrong addresses; all 8 were
                # unqualified either way (Issue 752).
                got["misattributed"] += 1
                tag = (f"  [⛔MISATTRIBUTED: names {'/'.join(sorted(bad))}, "
                       f"which does NOT own {n}]")
            elif cls is CROSS and (kind, n) in qualified_here:
                # The repair is MECHANICAL: this document already names the
                # owner for this number somewhere else, so the fix is to copy
                # that attribution here, with no lookup and no adjudication.
                # NOT a qualification — a bare number mid-document still
                # rebinds the day the repo allocates it locally, which is the
                # whole hazard. It ORDERS the work.
                got["repeat"] += 1
                tag = "  [repeat: this file attributes this number elsewhere]"
            elif cls is CROSS and hint:
                if hint & set(owners):
                    tag = f"  [crate-hint: {'/'.join(sorted(hint & set(owners)))}]"
                else:
                    got["misleading"] += 1
                    tag = (f"  [⛔MISLEADING: the only crate in the window is "
                           f"{'/'.join(sorted(hint))}, which does NOT own {n}]")
            if cls is CROSS:
                got["cross_units"].add((kind, n))
                # The COST of the lead-only alias rule, re-measured rather
                # than assumed: rows this sweep emits that a forward-reaching
                # alias would SUPPRESS (Issue 753). A triage quantity with the
                # same standing as the adjudication count — never a verdict,
                # because every one needs a line-by-line read.
                if icg.alias_trail_owners(lines[ln - 1], kind, n, sibs) & set(owners):
                    got["alias_trailing"] += 1
            got[cls].append(
                f"{doc}:{ln}  {kind} {n} -> "
                f"{'/'.join(owners) if owners else 'NO REPO IN THE WORKSPACE'}"
                f"{f' (local top {top[kind]})' if cls is IN_RANGE else ''}"
                f"{tag}{gap_note}\n"
                f"          {lines[ln - 1].strip()[:110]}")
    return got


def gate_says() -> tuple[int, int, int]:
    """Run the per-push gate and READ its numbers. The sweep re-states a
    quantity the gate owns; asserting beats trusting. -> (rc, scanned, findings)

    `scanned` is **-2** when the gate DEFERRED its cross-repo adjudication
    (Issue 793 T3). Under `DOCS_GATE_PARTIAL_CLONE=1` the gate prints
    `partial: N citations scanned in …` rather than `scanned N citations`, and
    a regex that knows only the second shape returns -1 — which this sweep
    reads as "the instrument is untrustworthy" and exits 2. That is a correct
    refusal reached for the wrong reason: the gate is fine and said so. -2 is
    a THIRD state, not folded into either neighbour, because "the gate could
    not be read" and "the gate declined to adjudicate" call for opposite
    responses.
    """
    # Both halves of Issue 778: `encoding=` pins OUR decode (text=True would
    # use the system locale and hand back mojibake, or None with rc intact),
    # and PYTHONIOENCODING pins the gate's own stdout encoder so its `✓`
    # survives the write on a non-UTF-8 box.
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"},)
    scanned = re.search(r"scanned (\d+) citations", r.stdout)
    failed = re.search(r"FAILED — (\d+) unqualified", r.stdout)
    deferred = re.search(r"partial-clone scope \(DOCS_GATE_PARTIAL_CLONE", r.stdout)
    if scanned is None and deferred:
        return (r.returncode, -2, 0)
    return (r.returncode,
            int(scanned.group(1)) if scanned else -1,
            int(failed.group(1)) if failed else (0 if r.returncode == 0 else -1))


def worktree_arms() -> list[str]:
    """Issue 797 — the worktree-vs-HEAD split, against a REAL git repository.

    A known-answer arm in BOTH directions, because either alone certifies the
    wrong thing: UNCOMMITTED alone would pass on an instrument that simply
    ignored HEAD, and MASKED alone would pass on one that ignored the worktree.
    The fixtures are real commits — a stubbed `head_text` would test the stub,
    and the `.git` probe is the part most likely to go wrong.
    """
    import subprocess
    import tempfile

    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    def git(cwd, *args):
        subprocess.run(["git", "-C", str(cwd), *args], check=True,
                       capture_output=True)

    CLEAN = "nothing to cite here\n"
    CITE = "a bare cross reference to Issue 500 with no owner named\n"

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        me, sib = ws / "fake-repo", ws / "riir-fakesib"
        for r in (me, sib):
            (r / ".issues").mkdir(parents=True)
            (r / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (me / ".issues" / "010_local.md").write_text("x", encoding="utf-8")
        (sib / ".issues" / "500_sib.md").write_text("x", encoding="utf-8")
        alloc = {"fake-repo": {k: ({10} if k == "Issue" else set())
                               for k in icg.KINDS},
                 "riir-fakesib": {k: ({500} if k == "Issue" else set())
                                  for k in icg.KINDS}}
        docs = ["AGENTS.md"]

        def run(read=_worktree_read):
            return audit(me, [sib], alloc, docs, {}, {}, read=read)

        # ── the COMMITTED state carries the finding ───────────────────────
        (me / "AGENTS.md").write_text(CITE, encoding="utf-8")
        git(me, "init", "-q", "-b", "main")
        git(me, "config", "user.email", "t@t")
        git(me, "config", "user.name", "t")
        git(me, "add", "-A")
        git(me, "commit", "-qm", "init")

        base = run()
        check(len(base[CROSS]) == 1,
              f"the fixture does not produce a CROSS row at all, so neither "
              f"direction below can mean anything: {base[CROSS]}")

        def head_read(p: Path) -> str | None:
            return head_text(me, p.name)

        # ── MASKED: the worktree HIDES a committed finding ─────────────────
        (me / "AGENTS.md").write_text(CLEAN, encoding="utf-8")
        wt, hd = run(), run(head_read)
        check(dirty_files(me) == {"AGENTS.md"},
              f"the dirty set did not see the edit: {dirty_files(me)}")
        check(len(wt[CROSS]) == 0 and len(hd[CROSS]) == 1,
              f"MASKED direction: worktree={len(wt[CROSS])} HEAD={len(hd[CROSS])} "
              f"— expected a clean worktree over a dirty HEAD")
        wt_keys = {_row_key(r) for c in _CLASSES for r in wt[c]}
        masked = [r for c in _CLASSES for r in hd[c] if _row_key(r) not in wt_keys]
        check(len(masked) == 1,
              f"the MASKED diff did not recover the committed row: {masked}")

        # ── UNCOMMITTED: the worktree INVENTS a finding HEAD does not have ──
        # Two citations, so the row count moves AND the line numbers shift —
        # the arm that reds if `_row_key` ever grows a line number back.
        (me / "AGENTS.md").write_text("padding line\n" + CITE + CITE,
                                      encoding="utf-8")
        wt, hd = run(), run(head_read)
        check(len(wt[CROSS]) == 2 and len(hd[CROSS]) == 1,
              f"UNCOMMITTED direction: worktree={len(wt[CROSS])} "
              f"HEAD={len(hd[CROSS])} — expected 2 over 1")
        wt_keys = {_row_key(r) for c in _CLASSES for r in wt[c]}
        hd_keys = {_row_key(r) for c in _CLASSES for r in hd[c]}
        check(wt_keys == hd_keys,
              f"_row_key is not line-free — a shifted line reported the SAME "
              f"citation as both UNCOMMITTED and MASKED: {wt_keys} vs {hd_keys}")

        # ── the injected reader must be able to say "not in HEAD" ──────────
        # A document added but never committed has no HEAD blob, and `None`
        # must skip it rather than read the worktree behind the caller's back.
        # Restored to HEAD's bytes, so the repo is clean again and the only
        # difference below is the document git has never heard of.
        (me / "AGENTS.md").write_text(CITE, encoding="utf-8")
        check(dirty_files(me) == frozenset(),
              f"restoring HEAD's bytes did not clean the repo: {dirty_files(me)}")
        (me / "NEW.md").write_text(CITE, encoding="utf-8")
        two = audit(me, [sib], alloc, ["AGENTS.md", "NEW.md"], {}, {})
        check(two["n_docs"] == 2 and len(two[CROSS]) == 2,
              f"the worktree read did not see the new document: {two['n_docs']}")
        hd2 = audit(me, [sib], alloc, ["AGENTS.md", "NEW.md"], {}, {},
                    read=lambda p: head_text(me, p.name))
        check(hd2["n_docs"] == 1 and len(hd2[CROSS]) == 1,
              f"a document absent from HEAD was not skipped — n_docs="
              f"{hd2['n_docs']}, cross={len(hd2[CROSS])}; a reader that fell "
              f"back to the worktree here would report a committed finding "
              f"that does not exist")
    return fails


# ── Issue 921 Arm B: the two-sided fixture ─────────────────────────────────
# The commit that REPAIRED the founding specimen (the bare `Issue 980` row in
# riir-infer HISTORY.md — the citation-drift sweep's own founding ILR row)
# and its parent. At the parent the row must classify CROSS -> riir-ai
# through the gap; at the fix it must find nothing. Numbers are FROZEN
# history, so the fixture's answer is independent of every working tree.
PROVE_REPO = "riir-infer"
PROVE_PARENT = "f0191e3^"
PROVE_FIX = "f0191e3"
PROVE_N = 980
PROVE_OWNER = "riir-ai"
# counter=35 (.highwater_local 035); floor=998 — the smallest full-history
# `.issues` ADD above 035 (998/1003/1004 are the inherited carve docs; 998
# was added and later REMOVED, which is exactly why the fixture below CLONES
# the repo instead of archive-extracting it: the floor leg is a git-log leg,
# and an extraction has no history — it would derive the tree's smallest
# remaining add and under-test the very derivation this arm pins).
PROVE_GAP = (35, 998)


def prove_fires() -> int:
    """Two-sided known-answer at the riir-infer fixtures (Issue 921 Arm B).

    Follows `platform_dead_code_audit.prove_fires`' shape — run the classifier
    over a tree whose answer is known independently, require BOTH sides, and
    refuse loudly when the fixture cannot be assembled — with one adaptation
    the family shape does not need: this classifier's floor leg reads
    `git log`, so the fixture is a LOCAL CLONE checked out at each rev
    (read-only against the source), not an archive extraction.

    `unreliable` is INJECTED EMPTY, exactly as the Issue-827 selftest arm
    injects its oracle: the fixture pins the CLASSIFIER's buckets, never this
    box's fetch schedule — a stale riir-ai checkout must not flip the
    expected verdict (staleness is that arm's subject, separately armed).
    """
    import tempfile

    repos = icg.contract_repos(WORKSPACE)
    src = next((r for r in repos if r.name == PROVE_REPO), None)
    if src is None:
        print(f"  ⚠ DEFERRED — {PROVE_REPO} is not in the derived population "
              f"on this box, so the fixture cannot be assembled (a loud "
              f"deferral, never a green: the gap branch is untested here)")
        return 0
    src_disk = repo_alias.real(src)
    if not src_disk.is_dir():
        print(f"  ⚠ DEFERRED — {PROVE_REPO}'s checkout is not on disk here")
        return 0
    for rev in (PROVE_PARENT, PROVE_FIX):
        v = subprocess.run(
            ["git", "-C", str(src_disk), "rev-parse", "--verify", "--quiet",
             f"{rev}^{{commit}}"], capture_output=True, encoding="utf-8")
        if v.returncode != 0:
            print(f"  ⛔ fixture commit {rev} is not reachable in "
                  f"{src_disk.name} — a box that HAS the repo must have the "
                  f"fixture; refusing rather than reporting a vacuous pass")
            return 2

    docs = icg.parse_pins(GATE_PINS)["documents"]
    assert isinstance(docs, list)
    crates = crate_map(repos)
    patterns = {c: re.compile(r"\b" + re.escape(c).replace(r"\-", "[-_]") + r"\b")
                for c in crates}
    alloc_base = {r.name: {k: icg.allocated(repo_alias.real(r), d)
                           for k, d in icg.KINDS.items()}
                  for r in repos if r.name != PROVE_REPO}
    sibs = [r for r in repos if r.name != PROVE_REPO]
    key = ("HISTORY.md", "Issue", str(PROVE_N))

    rc = 0
    with tempfile.TemporaryDirectory() as td:
        # The directory carries the ON-DISK spelling, so repo_alias.real()
        # resolves to it on every box; the audit HANDLE carries the CONTRACT
        # name, so alloc keys, owner names and stdout stay in the contract
        # vocabulary (the round-1 Hole-1 keying note).
        dest = Path(td) / repo_alias.disk(PROVE_REPO)
        c = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks",
             str(src_disk), str(dest)],
            capture_output=True, encoding="utf-8")
        if c.returncode != 0:
            print(f"  ⛔ fixture clone failed: {c.stderr.strip()}")
            return 2
        for rev, want_gap in ((PROVE_PARENT, True), (PROVE_FIX, False)):
            co = subprocess.run(
                ["git", "-C", str(dest), "checkout", "--quiet",
                 "--detach", rev], capture_output=True, encoding="utf-8")
            if co.returncode != 0:
                print(f"  ⛔ fixture checkout {rev} failed: {co.stderr.strip()}")
                return 2
            handle = Path(td) / PROVE_REPO
            alloc = {**alloc_base,
                     PROVE_REPO: {k: icg.allocated(dest, d)
                                  for k, d in icg.KINDS.items()}}
            got = audit(handle, sibs, alloc, docs, crates, patterns,
                        unreliable={})
            if got["n_cites"] == 0:
                print(f"  ⛔ {rev}: zero citations in the fixture — the walk "
                      f"went blind and the arm measured NOTHING")
                rc = 2
                continue
            g = got["gap"].get("Issue")
            rows = [r for c_ in _CLASSES for r in got[c_]
                    if _row_key(r) == key]
            cross = [r for r in got[CROSS] if _row_key(r) == key]
            tail = (f" — cites={got['n_cites']} ilr={len(got[IN_RANGE])} "
                    f"cross={len(got[CROSS])} orphan={len(got[ORPHAN])}")
            if want_gap:
                ok = (g == PROVE_GAP and len(rows) == 1 and len(cross) == 1
                      and PROVE_OWNER in cross[0])
                print(f"  {'✓' if ok else '✗'} {rev}: Issue {PROVE_N} "
                      f"{'CROSS -> ' + PROVE_OWNER if len(cross) == 1 else rows} "
                      f"[gap {g}]{tail}")
                if ok:
                    continue
                rc = rc or 1
                if g != PROVE_GAP:
                    print(f"      gap derived {g} != {PROVE_GAP} "
                          f"(counter 35; floor = the smallest full-history "
                          f".issues ADD above it = 998)")
                if len(cross) != 1 or PROVE_OWNER not in (cross[0] if cross else ""):
                    print(f"      expected exactly one CROSS row -> "
                          f"{PROVE_OWNER}; got {rows}")
            else:
                ok = not rows and g == PROVE_GAP
                print(f"  {'✓' if ok else '✗'} {rev}: Issue {PROVE_N} "
                      f"{'no finding (qualified)' if not rows else rows} "
                      f"[gap {g}]{tail}")
                if ok:
                    continue
                rc = rc or 1
                if rows:
                    print(f"      the repaired row must find NOTHING; got {rows}")
                if g != PROVE_GAP:
                    print(f"      gap derived {g} != {PROVE_GAP}")
    if rc:
        print("  ⛔ --prove-fires did not reproduce the known answer")
    else:
        print("  ✓ --prove-fires PASSED — both sides of the two-lane gap "
              "fixture fire as recorded")
    return rc


def selftest() -> list[str]:
    """Prove every bucket FIRES through THIS sweep's `audit()`, that the
    qualified control does NOT, and that the pin parser refuses a short row.
    Each fails silently otherwise, and a silent failure reports a clean
    workspace."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        me, sib = ws / "fake-repo", ws / "riir-fakesib"
        (me / ".issues").mkdir(parents=True)
        (sib / ".issues").mkdir(parents=True)
        (me / ".issues" / "010_local.md").write_text("x", encoding="utf-8")
        for n in ("010", "500", "600"):
            (sib / ".issues" / f"{n}_sib.md").write_text("x", encoding="utf-8")
        (me / "crates" / "thing").mkdir(parents=True)
        (me / "crates" / "thing" / "Cargo.toml").write_text('[package]\nname = "sibcrate-x"\n', encoding="utf-8")
        crates = {"sibcrate-x": "riir-fakesib"}
        pats = {c: re.compile(r"\b" + re.escape(c).replace(r"\-", "[-_]") + r"\b")
                for c in crates}
        alloc = {"fake-repo": {k: (set() if k != "Issue" else {10}) for k in icg.KINDS},
                 "riir-fakesib": {k: (set() if k != "Issue" else {10, 500, 600})
                                  for k in icg.KINDS}}

        (me / "AGENTS.md").write_text(
            "Issue 500 is the bare cross-repo row.\n"          # CROSS
            "Issue 006 was never filed here.\n"                # IN-LOCAL-RANGE (<= 10)
            "Issue 900 belongs to nobody at all.\n"            # ORPHAN
            "Issue 010 is local and the sibling has one.\n"    # AMBIGUOUS
            "riir-fakesib Issue 600 names its repo.\n"         # QUALIFIED (window)
            "`sibcrate-x` ships it; Issue 600 rides the crate.\n", encoding="utf-8")  # crate-hint
        got = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)

        if got["n_cites"] != 6:
            fails.append(f"citation walk found {got['n_cites']}, expected 6 — "
                         f"every arm below is vacuous")
        for cls, want in ((CROSS, 1), (IN_RANGE, 1), (ORPHAN, 1)):
            if len(got[cls]) != want:
                fails.append(f"{cls}: got {len(got[cls])} rows, expected {want}: {got[cls]}")
        if len(got["ambiguous"]) != 1:
            fails.append(f"AMBIGUOUS: got {got['ambiguous']}, expected 1")
        if got[CROSS] and "500" not in got[CROSS][0]:
            fails.append(f"CROSS picked the wrong row: {got[CROSS][0]}")

        # CONTROL: a qualified citation must produce NO finding, or the sweep
        # reds on every correct repair and gets switched off.
        (me / "AGENTS.md").write_text("riir-fakesib Issue 500 is qualified.\n", encoding="utf-8")
        ctl = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if ctl[CROSS] or ctl[IN_RANGE] or ctl[ORPHAN]:
            fails.append(f"control: a QUALIFIED citation produced a finding: {ctl}")
        if ctl["n_cites"] != 1:
            fails.append("control: the qualified citation left the population")

        # the two crate sub-classes, and they are OPPOSITE verdicts. A second
        # sibling is required: with one sibling every crate hit trivially names
        # the owner, and MISLEADING could never be constructed — the arm would
        # pass while testing nothing (a green canary that cannot fire).
        other = ws / "riir-otherlib"
        (other / ".issues").mkdir(parents=True)
        alloc["riir-otherlib"] = {k: set() for k in icg.KINDS}
        crates2 = dict(crates, **{"othercrate-y": "riir-otherlib"})
        pats2 = {c: re.compile(r"\b" + re.escape(c).replace(r"\-", "[-_]") + r"\b")
                 for c in crates2}
        (me / "AGENTS.md").write_text("`sibcrate-x` ships it; Issue 500 rides the crate.\n", encoding="utf-8")
        hint = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2)
        if len(hint[CROSS]) != 1 or "crate-hint" not in hint[CROSS][0]:
            fails.append(f"crate-hint sub-class did not fire: {hint[CROSS]}")
        if hint["misleading"]:
            fails.append("crate naming the OWNER was counted as misleading")

        (me / "AGENTS.md").write_text("`othercrate-y` moved it; Issue 500 is elsewhere.\n", encoding="utf-8")
        mis = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2)
        if mis["misleading"] != 1 or len(mis[CROSS]) != 1:
            fails.append(f"MISLEADING sub-class did not fire: {mis['misleading']} "
                         f"{mis[CROSS]}")

        # ── Issue 827: the ORACLE's freshness is part of the verdict ────────
        # The measured shape, exactly: the citation NAMES a repo, that repo
        # really owns the number, and this box's checkout of it has not seen
        # the allocation — so `owners` lists somebody ELSE and the named repo
        # reads as a wrong address. `riir-fakesib` owns 500 here, so naming
        # `riir-otherlib` is the accusation; injecting `unreliable` is what
        # says "we cannot believe that answer".
        #
        # `unreliable` is INJECTED rather than derived, so the arm tests the
        # RULE and not this box's fetch schedule — the whole defect was a
        # verdict that moved with a checkout's age.
        (me / "AGENTS.md").write_text("riir-otherlib Issue 500 moved there.\n", encoding="utf-8")

        # (a) oracle believed current -> the ordinary accusation. Without this
        #     side the rule could suppress everything and still pass.
        fresh = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2)
        if not fresh[CROSS] or not fresh["misattributed"]:
            fails.append(f"827 control: the fixture did not produce the "
                         f"accusation it must suppress — cross={fresh[CROSS]} "
                         f"misat={fresh['misattributed']}")
        if fresh[ORACLE_STALE]:
            fails.append(f"827: a CURRENT oracle was called stale: "
                         f"{fresh[ORACLE_STALE]}")

        # (b) the SAME text, that oracle unreliable -> UNDECIDED. Not merely
        #     unlabelled: it must leave the COUNTED buckets. The measured case
        #     breached a ceiling, and suppressing the ⛔ tag alone would have
        #     left the red standing.
        st = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2,
                   unreliable={"riir-otherlib": "260 commits behind upstream"})
        if len(st[ORACLE_STALE]) != 1:
            fails.append(f"827: a stale oracle did not produce an UNDECIDED "
                         f"row: {st[ORACLE_STALE]}")
        if st[CROSS] or st[ORPHAN] or st[IN_RANGE] or st[MISATTR_IN_RANGE]:
            fails.append(
                f"827: a stale-oracle row was still COUNTED — cross="
                f"{st[CROSS]} orphan={st[ORPHAN]} in_range={st[IN_RANGE]} "
                f"wrong_addr={st[MISATTR_IN_RANGE]}; the ceiling stays "
                f"breachable by another repo's fetch schedule")
        if st["misattributed"] or st["misleading"]:
            fails.append("827: a stale oracle still produced an accusation "
                         "sub-count")
        if st["n_cites"] != fresh["n_cites"]:
            fails.append("827: the citation left the POPULATION — UNDECIDED "
                         "must not shrink the denominator")

        # (c) staleness in a repo the citation does NOT name changes nothing.
        #     Without this the rule could suppress on any staleness anywhere.
        un = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2,
                   unreliable={"riir-somebody-else": "never fetched"})
        if un[ORACLE_STALE]:
            fails.append(f"827: staleness in an UNNAMED repo suppressed a row: "
                         f"{un[ORACLE_STALE]}")
        if not un[CROSS]:
            fails.append("827: an unrelated stale repo suppressed the finding")

        # ── the two RULE-COST probes must FIRE, and their controls must NOT
        # (Issue 753). Both report a 0 in the live workspace, which is exactly
        # the shape a probe wired to nothing also reports.
        (me / "AGENTS.md").write_text(
            "## Bench 1: a section heading, not a citation\n"
            "Issue 500 is the real one.\n", encoding="utf-8")
        w = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if w["unseen_width"] != 1:
            fails.append(f"width complement did not fire: {w['unseen_width']} != 1")
        if w["n_cites"] != 1:
            fails.append(f"width: `Bench 1` must NOT enter the walk, got "
                         f"{w['n_cites']} citations")
        (me / "AGENTS.md").write_text("Issue 500 alone.\n", encoding="utf-8")
        if audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)["unseen_width"]:
            fails.append("width complement counted a 3-digit citation")

        # an alias AFTER the citation: still CROSS (the rule is lead-only), and
        # counted as the cost of that decision.
        (me / "AGENTS.md").write_text("Issue 500, over in fakesib somewhere.\n", encoding="utf-8")
        tr = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if len(tr[CROSS]) != 1:
            fails.append(f"a TRAILING alias must not qualify: {tr[CROSS]}")
        if tr["alias_trailing"] != 1:
            fails.append(f"alias-trailing cost did not fire: {tr['alias_trailing']}")
        # CONTROL A: the same alias in the LEAD qualifies, so no row and no cost.
        (me / "AGENTS.md").write_text("fakesib Issue 500 is addressed.\n", encoding="utf-8")
        lead = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if lead[CROSS] or lead["alias_trailing"]:
            fails.append(f"lead alias must qualify: {lead[CROSS]} {lead['alias_trailing']}")
        # CONTROL B: a trailing alias of a NON-owner is not a suppression cost.
        (me / "AGENTS.md").write_text("Issue 500, over in otherlib somewhere.\n", encoding="utf-8")
        if audit(me, [sib, other], alloc, ["AGENTS.md"], crates, pats)["alias_trailing"]:
            fails.append("alias-trailing counted a NON-owner alias")

        # ── Issue 846: the ON-DISK spellings of the aliased repos ─────────
        # The contract names mmorpg-editor / mmorpg-remake (mmorpg-remaster retired 2026-11-12)
        # exist only in repo_set.txt; the directories are on disk as
        # seal-game-editor / seal-remake on every box (seal-online-remaster retired 2026-11-12)
        # measured, and the docs name them that way. A spelling qualifies via
        # the lead OR the window (which reads the citation's own line
        # forward), never accuses, and a longer name does not match. The
        # fixture resolves its files through repo_alias.disk() so the arms
        # are box-independent: identity on a clean box, the aliased spelling
        # on one that carries the mapping.
        import repo_alias as ra
        alias_disk = ws / ra.disk("mmorpg-remake")
        (alias_disk / ".issues").mkdir(parents=True, exist_ok=True)
        (alias_disk / ".issues" / "011_spelled.md").write_text("x", encoding="utf-8")
        alias_sib = ws / "mmorpg-remake"          # the CONTRACT handle
        alloc["mmorpg-remake"] = {k: (set() if k != "Issue" else {11})
                                  for k in icg.KINDS}

        (me / "AGENTS.md").write_text("seal-remake Issue 011 is addressed.\n", encoding="utf-8")
        sp = audit(me, [sib, alias_sib], alloc, ["AGENTS.md"], crates, pats)
        if sp[CROSS] or sp["misattributed"]:
            fails.append(f"846: a spelling ON the lead must qualify: "
                         f"{sp[CROSS]} {sp['misattributed']}")

        (me / "AGENTS.md").write_text("authored at seal-remake:\nIssue 011 came from there.\n", encoding="utf-8")
        spw = audit(me, [sib, alias_sib], alloc, ["AGENTS.md"], crates, pats)
        if spw[CROSS]:
            fails.append(f"846: a spelling in the WINDOW must qualify: {spw[CROSS]}")

        (me / "AGENTS.md").write_text("Issue 011 landed in seal-remake later.\n", encoding="utf-8")
        spt = audit(me, [sib, alias_sib], alloc, ["AGENTS.md"], crates, pats)
        if spt[CROSS]:
            fails.append(f"846: a spelling TRAILING on the citation's own line "
                         f"must qualify through the window: {spt[CROSS]}")
        if spt["alias_trailing"]:
            fails.append("846: a spelling must not count as alias-trailing "
                         "cost — the window already settles it")

        (me / "AGENTS.md").write_text("seal-remake-unity Issue 011 is the longer name.\n", encoding="utf-8")
        spb = audit(me, [sib, alias_sib], alloc, ["AGENTS.md"], crates, pats)
        if len(spb[CROSS]) != 1:
            fails.append(f"846: seal-remake-unity must NOT name seal-remake: {spb[CROSS]}")

        del alloc["mmorpg-remake"]

        # padding is the SAME number (Issue 751 T2b) — `006` must read as 6
        (me / "AGENTS.md").write_text("Issue 0500 no; Issue 500 yes.\n", encoding="utf-8")
        pad = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if len(pad[CROSS]) != 2:
            fails.append(f"zero-padding: `Issue 0500` and `Issue 500` must be "
                         f"ONE number, got {len(pad[CROSS])} CROSS rows")

        # ── Issue 794: MISATTRIBUTED-IN-RANGE, and the boundary that keeps it
        # honest. Four arms, because the class is defined as much by what it
        # must NOT promote as by what it must. A third sibling is required:
        # the number has to be IN this repo's range, unallocated here, and
        # owned by a repo the prose does NOT name — one sibling cannot build
        # that (`is_qualified` accepts any named repo when NOTHING owns the
        # number, which is the ORPHAN rule, not this one).
        third = ws / "riir-thirdlib"
        (third / ".issues").mkdir(parents=True)
        (third / ".issues" / "006_theirs.md").write_text("x", encoding="utf-8")
        alloc["riir-thirdlib"] = {k: (set() if k != "Issue" else {6})
                                  for k in icg.KINDS}
        s3 = [sib, third]
        (me / "AGENTS.md").write_text("riir-fakesib Issue 006 is the wrong address.\n", encoding="utf-8")
        wa = audit(me, s3, alloc, ["AGENTS.md"], crates, pats)
        if len(wa[MISATTR_IN_RANGE]) != 1:
            fails.append(f"MISATTRIBUTED-IN-RANGE did not fire: "
                         f"{wa[MISATTR_IN_RANGE]} / {wa[IN_RANGE]}")
        if len(wa[IN_RANGE]) != 1 or "MISATTRIBUTED-IN-RANGE" not in wa[IN_RANGE][0]:
            fails.append(f"the row must stay in IN_RANGE and carry its tag: "
                         f"{wa[IN_RANGE]}")
        if wa[CROSS]:
            fails.append(f"MISATTRIBUTED-IN-RANGE must not also count as CROSS: "
                         f"{wa[CROSS]} — the two repairs read differently")

        # CONTROL A: the named repo OWNS it -> not a finding at all.
        (me / "AGENTS.md").write_text("riir-thirdlib Issue 006 is addressed.\n", encoding="utf-8")
        ca = audit(me, s3, alloc, ["AGENTS.md"], crates, pats)
        if ca[MISATTR_IN_RANGE] or ca[IN_RANGE] or ca[CROSS] or ca[ORPHAN]:
            fails.append(f"control A: a correctly-addressed in-range citation "
                         f"produced a finding: {ca}")

        # CONTROL B: bare, no attribution -> still UNDECIDED, never promoted.
        (me / "AGENTS.md").write_text("Issue 006 is bare.\n", encoding="utf-8")
        cb = audit(me, s3, alloc, ["AGENTS.md"], crates, pats)
        if cb[MISATTR_IN_RANGE] or len(cb[IN_RANGE]) != 1:
            fails.append(f"control B: a bare in-range citation must stay "
                         f"UNDECIDED: {cb[MISATTR_IN_RANGE]} / {cb[IN_RANGE]}")

        # CONTROL C: the repo name is in the 3-line WINDOW but not ON the
        # citation. `adj` is lead-only by design (Issue 752) and the promotion
        # inherits that — a name that was never an attribution must not become
        # a wrong address.
        (me / "AGENTS.md").write_text("riir-fakesib ships other things.\n"
                                      "Issue 006 is bare here.\n", encoding="utf-8")
        cc = audit(me, s3, alloc, ["AGENTS.md"], crates, pats)
        if cc[MISATTR_IN_RANGE]:
            fails.append(f"control C: a WINDOW-only repo name must not promote "
                         f"an in-range row: {cc[MISATTR_IN_RANGE]}")

        # CONTROL D: the MEASURED exemption (19 rows, 19 false). A number this
        # repo DID allocate, with a non-owner sibling named right on it — the
        # prose contrasting a local number with a remote one. Must stay
        # AMBIGUOUS, promoted by nothing.
        (me / "AGENTS.md").write_text("riir-thirdlib Issue 010 is this repo's own.\n", encoding="utf-8")
        cd = audit(me, s3, alloc, ["AGENTS.md"], crates, pats)
        if cd[MISATTR_IN_RANGE] or cd[IN_RANGE] or cd[CROSS]:
            fails.append(f"control D: a LOCALLY-ALLOCATED number must not be "
                         f"promoted by an adjacent non-owner name — that is "
                         f"the 19/19-false exemption: {cd}")
        if len(cd["ambiguous"]) != 1:
            fails.append(f"control D: the row must remain AMBIGUOUS: {cd['ambiguous']}")

        # ── Issue 921 Arm B: the two-lane gap via .highwater_local ──────────
        # `gp` is its OWN fixture (not `me`) so the git-dance phase below
        # cannot leak repo state into the arms above. Membership first:
        # WITHOUT the file the branch is dead and the classification is the
        # pre-921 one, byte for byte.
        gp = ws / "riir-gaprepo"
        (gp / ".issues").mkdir(parents=True)
        (gp / ".issues" / "010_local.md").write_text("x", encoding="utf-8")
        (gp / ".issues" / ".highwater").write_text("600", encoding="utf-8")
        alloc["riir-gaprepo"] = {k: (set() if k != "Issue" else {10})
                                 for k in icg.KINDS}

        (gp / "AGENTS.md").write_text("Issue 500 bare, no local lane.\n",
                                      encoding="utf-8")
        nb = audit(gp, [sib], alloc, ["AGENTS.md"], crates, pats)
        if nb["gap"] or len(nb[IN_RANGE]) != 1 or nb[CROSS] or nb[ORPHAN]:
            fails.append(f"921 membership: without .highwater_local the "
                         f"classification must be the pre-921 one (gap empty, "
                         f"Issue 500 IN-LOCAL-RANGE): gap={nb['gap']} "
                         f"cross={nb[CROSS]} ilr={nb[IN_RANGE]}")

        (gp / ".issues" / ".highwater_local").write_text("010", encoding="utf-8")
        (gp / "AGENTS.md").write_text(
            "Issue 500 is inherited-only.\n"          # 10 < 500, floor open -> CROSS
            "Issue 009 is under the counter.\n"       # 9 <= 10 -> IN-LOCAL-RANGE
            "Issue 010 is allocated here.\n"          # local (ambiguous w/ sib)
            "Issue 011 is one above the counter.\n",  # in gap, unowned -> ORPHAN
            encoding="utf-8")
        gp2 = audit(gp, [sib], alloc, ["AGENTS.md"], crates, pats)
        if gp2["gap"] != {"Issue": (10, None)}:
            fails.append(f"921: derived gap {gp2['gap']} != "
                         f"{{'Issue': (10, None)}}")
        if (len(gp2[CROSS]) != 1 or "500" not in gp2[CROSS][0]
                or "riir-fakesib" not in gp2[CROSS][0]
                or "two-lane gap" not in gp2[CROSS][0]):
            fails.append(f"921: an open-floor gap row must be CROSS -> "
                         f"riir-fakesib carrying the gap note: {gp2[CROSS]}")
        if len(gp2[IN_RANGE]) != 1 or "009" not in gp2[IN_RANGE][0]:
            fails.append(f"921: BELOW the counter stays UNDECIDED-local: "
                         f"{gp2[IN_RANGE]}")
        if len(gp2[ORPHAN]) != 1 or "11" not in gp2[ORPHAN][0]:
            fails.append(f"921: counter<n<floor with no owner must be ORPHAN "
                         f"— actionable, never UNDECIDED: {gp2[ORPHAN]}")
        if gp2["ambiguous"] != {("Issue", 10)}:
            fails.append(f"921: the locally-allocated 010 must stay local "
                         f"(AMBIGUOUS with the sibling's 10): {gp2['ambiguous']}")

        # Phase 2: a BOUNDED floor needs real git-log ADDS — the same record
        # source `file_and_history_allocated` reads. The fixture is committed
        # so the temp tree answers the git-log leg deterministically.
        (gp / ".issues" / "050_floor_edge.md").write_text("x", encoding="utf-8")

        def _g(*args):
            subprocess.run(["git", "-C", str(gp), *args], check=True,
                           capture_output=True)

        _g("init", "-q", "-b", "main")
        _g("config", "user.email", "t@t")
        _g("config", "user.name", "t")
        _g("add", "-A")
        _g("commit", "-qm", "fx")
        (gp / "AGENTS.md").write_text(
            "Issue 500 is at or above the floor.\n"   # 500 >= 50 -> IN-RANGE again
            "Issue 030 rides the gap.\n",             # 10 < 30 < 50 -> ORPHAN
            encoding="utf-8")
        gp3 = audit(gp, [sib], alloc, ["AGENTS.md"], crates, pats)
        if gp3["gap"] != {"Issue": (10, 50)}:
            fails.append(f"921: the git-log floor {gp3['gap']} != "
                         f"{{'Issue': (10, 50)}}")
        if len(gp3[IN_RANGE]) != 1 or "500" not in gp3[IN_RANGE][0]:
            fails.append(f"921: the gap must STOP at the floor (500 >= 50 "
                         f"stays IN-LOCAL-RANGE): {gp3[IN_RANGE]} / {gp3[CROSS]}")
        if len(gp3[ORPHAN]) != 1 or "30" not in gp3[ORPHAN][0]:
            fails.append(f"921: inside the bounded gap must be actionable: "
                         f"{gp3[ORPHAN]}")

        # Phase 3: heading allocations are EXCLUDED from the floor — an
        # Arm-A-style heading addition must not shrink the gap
        # (order-dependence). The heading DOES allocate 30 and the arm proves
        # that below, so the exclusion is deliberate and not oracle blindness;
        # a number BETWEEN the heading and the floor discriminates.
        (gp / "HISTORY.md").write_text(
            "## Issue 030 (2026-01-01) — a heading inside the gap\n",
            encoding="utf-8")
        alloc["riir-gaprepo"]["Issue"] = {10, 30}
        (gp / "AGENTS.md").write_text(
            "Issue 040 sits between heading and floor.\n", encoding="utf-8")
        gp4 = audit(gp, [sib], alloc, ["AGENTS.md"], crates, pats)
        if gp4["gap"] != {"Issue": (10, 50)}:
            fails.append(f"921: a heading allocation must NOT shrink the "
                         f"floor (heading additions are order-dependent): "
                         f"{gp4['gap']}")
        if len(gp4[ORPHAN]) != 1 or "40" not in gp4[ORPHAN][0]:
            fails.append(f"921: 40 must stay inside the gap: {gp4[ORPHAN]}")
        if 30 not in icg.heading_allocated(gp, ".issues", ["riir-gaprepo"]):
            fails.append("921: the heading-exclusion arm is dishonest — the "
                         "heading oracle does not even see 30, so the arm "
                         "proves blindness, not exclusion")
        del alloc["riir-gaprepo"]

        # pin parser: globals + 5-field rows, comments stripped, arity enforced
        pins = ws / "pins.txt"
        pins.write_text("# c\nmin_repos = 15\nrepo-a 10 0 0 0  # trailing\n\n", encoding="utf-8")
        g, rows = parse_pins(pins)
        if g != {"min_repos": 15} or rows != {"repo-a": dict(zip(FIELDS, (10, 0, 0, 0)))}:
            fails.append(f"pin parse: got {g} {rows}")
        pins.write_text("repo-a 1 2\n", encoding="utf-8")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass

        # ── Issue 823 T6: the width bound's own floor is READ from the pin
        # file and is a real global, not a default nobody set. A floor that is
        # silently absent is the same as no floor, and this one exists because
        # the quantity it guards fails by looking PERFECT.
        live_glob, _ = parse_pins(PINS)
        if "min_heading_shaped" not in live_glob:
            fails.append("min_heading_shaped is not pinned — the heading meter "
                         "has no blindness detector, and its blind output "
                         "(0/0) reads as perfect coverage")
        elif live_glob["min_heading_shaped"] < 1:
            fails.append("min_heading_shaped pinned at 0 or less — a floor that "
                         "cannot fail")

        # ── heading-only allocations (Issue 754). The one path in this
        # instrument that can SUPPRESS a finding, so all four arms are pinned:
        # the positive must fire, and each of the three measured negatives
        # must not — a heading rule that accepts `## Issue 043 follow-up (…)`
        # would absolve a wrong address, which is the defect, not the repair.
        hd = ws / "riir-headrepo"
        (hd / ".issues").mkdir(parents=True)
        (hd / "HISTORY.md").write_text(
            "## Issue 042 (2026-01-01) — resolved, file never committed\n"
            "## Issue 043 follow-up (2026-01-01) — about a FOREIGN number\n"
            "## Issue 044 (riir-fakesib) — an explicit foreign owner\n"
            "# Issue 045 (2026-01-01) — H1, a document title\n"
            "## Plan 046 (2026-01-01) — a different KIND\n"
            # Issue 828: the delimiter axis, on the SAME fixture rather
            # than a second one. 047 is the 56-record house style the
            # oracle could not spell; 048 is this issue's load-bearing
            # negative (the discriminator must survive the NEW delimiter,
            # or this IS the widening AGENTS.md calls unsound); 049 is a
            # live riir-ai shape where the ASCII hyphen is inside a WORD;
            # 050 is the dated form's `(`, which the leading form always
            # accepted -- an asymmetry between two patterns documented as
            # the same rule at two positions.
            "## Issue 047 — the dash delimiter, nothing interstitial\n"
            "## Issue 048 follow-up — commentary, NOT an allocation\n"
            "## Issue 049-class — the hyphen is inside a word\n"
            "## 2026-01-01 — Issue 050 (a parenthetical): the dated form\n", encoding="utf-8")
        names = ["riir-headrepo", "riir-fakesib"]
        # Issue 781 T2: the style blind spot is a PROBE, and a probe wired to
        # nothing reports 0 exactly like a clean tree (Issue 753). Both
        # directions over the SAME fixture: 042 is the style the oracle reads,
        # 043 is the style it rejects, and 044 is rejected for NAMING a
        # sibling — which must NOT be counted as a style loss, or the quantity
        # stops meaning what its label says.
        acc, shp = icg.heading_style_blind(hd, ".issues", names)
        if (acc, shp) != (3, 6):
            fails.append(f"heading style blind spot: got {(acc, shp)}, expected "
                         f"(3, 6) — 042/047/050 read, 043/048/049 are the "
                         f"style loss, 044 is a FOREIGN-name rejection and is "
                         f"excluded from both")

        got_h = icg.heading_allocated(hd, ".issues", names)
        if got_h != {42, 47, 50}:
            fails.append(f"heading allocation: got {sorted(got_h)}, expected "
                         f"[42, 47, 50] — 43/44/45/48/49 are the measured "
                         f"negatives, 46 is a Plan")
        if icg.heading_allocated(hd, ".plans", names) != {46}:
            fails.append("heading allocation: the KIND is not read from the subdir")
        # Issue 828 T4: the COST meter, both directions on the SAME fixture.
        # It is the quantity that decides whether the `resolved --` family is
        # worth an unsound rule, so a meter that always says 0 would retire
        # the question by looking like an answer.
        nov = icg.heading_unread_novel(hd, ".issues", names, known=set())
        if nov != 3:
            fails.append(f"heading unread COST: got {nov}, expected 3 -- "
                         f"43/48/49 are unread and no other oracle knows them")
        nov = icg.heading_unread_novel(hd, ".issues", names, known={43, 48, 49})
        if nov != 0:
            fails.append(f"heading unread COST: got {nov} with every unread "
                         f"number already KNOWN -- a record another oracle "
                         f"covers cannot change a verdict and must not be "
                         f"priced as if it could")
        # The ACCEPTED ones are never priced: 42/47/50 are already in the
        # owners set, so counting them would inflate the blast radius by
        # exactly the records the rule change does not touch.
        nov = icg.heading_unread_novel(hd, ".issues", names, known={43, 48})
        if nov != 1:
            fails.append(f"heading unread COST: got {nov}, expected 1 -- only "
                         f"the still-unknown unread record prices anything")

        if icg.allocated(hd, ".issues", names) != {42, 47, 50}:
            fails.append("allocated() does not union the heading path")

        # ── fenced headings are QUOTED, not allocated. Same standing as the
        # four arms above: a false allocation SUPPRESSES a finding, and the
        # shapes are the ones a markdown scanner actually gets wrong — an
        # inner ```bash must NOT close the block (a naive toggle then scans
        # the complement, reading prose as code to EOF), and a longer inner
        # run must not either.
        fz = ws / "riir-fencerepo"
        (fz / ".issues").mkdir(parents=True)
        # Fixture IO is pinned UTF-8 — pathlib text IO defaults to the LOCALE
        # codec, so an unpinned em-dash fixture round-trips on a Thai-locale
        # box (cp874 carries U+2014) but the same bytes read as UTF-8 elsewhere
        # diverge. The selftest must not be locale-dependent by construction.
        (fz / "HISTORY.md").write_text(
            "## Issue 042 (2026-01-01) — a REAL local allocation\n"
            "```markdown\n"
            "## Issue 043 (2026-01-01) — quoted from another repo's doc\n"
            "```bash\n"
            "## Issue 044 (2026-01-01) — still inside: an INFO STRING never closes\n"
            "````\n"
            "## Issue 045 (2026-01-01) — a longer BARE run does close it\n"
            "````python\n"
            "## Issue 047 (2026-01-01) — inside a 4-backtick block\n"
            "```\n"
            "## Issue 048 (2026-01-01) — a SHORTER run cannot close it\n"
            "````\n"
            "## Issue 049 (2026-01-01) — closed at equal width, a REAL allocation\n", encoding="utf-8")
        got_f = icg.heading_allocated(fz, ".issues", ["riir-fencerepo"])
        if got_f != {42, 45, 49}:
            fails.append(f"fenced headings: got {sorted(got_f)}, expected [42, 45, 49] "
                         f"— 43/44/47/48 are inside a fence and are QUOTED, not allocated")

        # the fail-SAFE arm: an unterminated fence excludes NOTHING (rather
        # than swallowing the tail to EOF, which is the same suppression one
        # level over) and is reported as its own hazard.
        (fz / "AGENTS.md").write_text(
            "```text\n"
            "## Issue 046 (2026-01-01) — inside an UNTERMINATED fence\n",
            encoding="utf-8")
        inside, open_at = icg.fenced_lines(
            (fz / "AGENTS.md").read_text(encoding="utf-8"))
        if inside or open_at != 0:
            fails.append(f"unterminated fence: got inside={sorted(inside)} open_at={open_at}, "
                         f"expected an EMPTY exclusion set and open_at=0")
        if 46 not in icg.heading_allocated(fz, ".issues", ["riir-fencerepo"]):
            fails.append("unterminated fence does not fail SAFE — the tail was "
                         "excluded, which suppresses allocations to EOF")
        if [d for d, _ in icg.unterminated_fences(fz)] != ["AGENTS.md"]:
            fails.append("unterminated_fences() does not report the hazard it creates")

        # population derivation: BOUNDARY.md + a .git DIRECTORY, both required
        (me / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (me / ".git").mkdir()
        (sib / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (sib / ".git").write_text("gitdir: elsewhere", encoding="utf-8")   # worktree-shaped
        if [p.name for p in icg.contract_repos(ws)] != ["fake-repo"]:
            fails.append(f"population derivation wrong: "
                         f"{[p.name for p in icg.contract_repos(ws)]}")
    return fails + worktree_arms()


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior

    fails = selftest()
    if fails:
        print("✗ citation sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    if "--prove-fires" in sys.argv:
        print("citation sweep — --prove-fires (the Issue-921 two-lane gap "
              "fixtures, riir-infer f0191e3^ vs f0191e3)")
        return prove_fires()

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        glob, pins = parse_pins(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    # ── T4: the quantities the per-push gate OWNS are asserted, not trusted ──
    gate_pins = icg.parse_pins(GATE_PINS)
    docs = gate_pins["documents"]
    assert isinstance(docs, list)
    if glob.get("min_repos") != gate_pins["min_repos"]:
        print(f"✗ pin drift: {PINS.name} says min_repos={glob.get('min_repos')}, "
              f"{GATE_PINS.name} says {gate_pins['min_repos']}. Same quantity, "
              f"two files — change both.")
        return 1

    repos = icg.contract_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2
    if len(repos) < glob["min_repos"]:
        print(f"✗ INSTRUMENT: derived {len(repos)} contract repos < floor "
              f"{glob['min_repos']} — the population went blind; every ceiling "
              f"below would pass vacuously")
        return 2

    # Issue 794. A MISSING ceiling is refused, never defaulted: a wall that
    # silently reads as "absent, so anything passes" is the green-zero shape
    # this whole family exists to refuse.
    if "max_misattributed_in_range" not in glob:
        print(f"✗ INSTRUMENT: {PINS.name} declares no "
              f"`max_misattributed_in_range` — the Issue 794 class would have "
              f"no ceiling and every row would pass silently")
        return 2
    glob_wall = glob["max_misattributed_in_range"]

    crates = crate_map(repos)
    patterns = {c: re.compile(r"\b" + re.escape(c).replace(r"\-", "[-_]") + r"\b")
                for c in crates}
    # Issue 842: allocations and the heading-oracle cost read the ON-DISK
    # directories, keyed by the CONTRACT name of the handle.
    alloc = {r.name: {k: icg.allocated(repo_alias.real(r), d)
                      for k, d in icg.KINDS.items()}
             for r in repos}
    # Issue 781: how much of each repo's own allocation record the heading
    # oracle declines to read, ON STYLE ALONE. Summed over kinds; a triage
    # quantity with the standing of AMBIGUOUS, never a verdict and never
    # folded into a finding count. Printed because the direction that HURTS is
    # currently 0 (an incomplete OWNERS set manufactures a FALSE
    # ⛔MISATTRIBUTED — Issue 754's failure, inherited by Issue 794's
    # in-range class), and a latent cost that is only remembered is one that
    # gets forgotten.
    #
    # Issue 828 T4 prices it. A record whose number is ALREADY known from a
    # file - in the worktree or in `git log` - contributes nothing whichever
    # way the rule goes, so only the residue can change a verdict and the
    # residue IS the blast radius. `novel` is that residue, and it is what
    # decides Issue 823 T5's open question: the unread COUNT looks like a
    # backlog and the unread COST is two rows workspace-wide.
    blind = {}
    for r in repos:
        acc = shp = nov = 0
        for k, d in icg.KINDS.items():
            a, t = icg.heading_style_blind(repo_alias.real(r), d,
                                           [q.name for q in repos])
            acc += a
            shp += t
            # `alloc` holds the FULL union (heading path included), so the
            # non-heading half is recomputed rather than subtracted: a number
            # in both halves must not be credited to the heading oracle, and
            # a set difference cannot tell the two apart.
            nov += icg.heading_unread_novel(repo_alias.real(r), d,
                                            [q.name for q in repos])
        blind[r.name] = (acc, shp, nov)

    # Issue 827: which sibling checkouts cannot be trusted to answer "do you
    # own this number?" — computed ONCE for the whole run, because it is a
    # property of the oracle repos and not of the repo being audited.
    unreliable = unreliable_oracles(repos)

    bad = False
    tot = {"docs": 0, "cites": 0, "amb": 0, "mis": 0, "misat": 0,
           "units": 0, "rep": 0, "width": 0, "trail": 0, MISATTR_IN_RANGE: 0,
           CROSS: 0, IN_RANGE: 0, ORPHAN: 0, ORACLE_STALE: 0}
    mine_row = None
    dirty_scope: dict[str, int] = {}
    n_uncommitted = n_masked = 0
    for repo in repos:
        # Issue 842: the handle is CONTRACT-named, the directory on-disk. All
        # git/filesystem reads go through `repo_disk`; every NAME (alloc keys,
        # pins, printouts) stays the contract spelling.
        repo_disk = repo_alias.real(repo)
        sibs = [s for s in repos if s != repo]
        # A repo is never its own oracle — `n in mine` short-circuits first —
        # so its own staleness is irrelevant here and is filtered out rather
        # than left to confuse the disclosure line.
        got = audit(repo, sibs, alloc, docs, crates, patterns,
                    unreliable={k: v for k, v in unreliable.items()
                                if k != repo.name})

        # ── Issue 797: the worktree is not the repo ──────────────────────────
        # This workspace runs concurrent sessions against SHARED worktrees, so
        # a row here may sit on a line no commit contains. Measured 2026-09-15:
        # the workspace's ENTIRE standing CROSS finding — 1 of 1 — was an
        # uncommitted edit stripping a `riir-train` qualifier HEAD carries.
        #
        # The split of responsibility: the DISPLAY shows the worktree (that is
        # what you see if you open the file, and it is what `gate_says()` reads,
        # so the T4 cross-check below must stay worktree-vs-worktree), while the
        # PINS adjudicate HEAD. A tracked expectations file is a claim about a
        # repo, and a repo's state is its commits — a ceiling re-pinned against
        # somebody's in-flight edit reds on every other box.
        dirty = dirty_files(repo_disk)
        scope = sorted(set(dirty) & set(docs))
        # Issue 921 Arm B: the two-lane counter moves the gap this sweep
        # classifies against, so it is IN this sweep's population — a dirty
        # counter discloses on the advisory below instead of silently
        # re-verdicting (the same standing as any other verdict-moving file).
        hw_dirty = HW_LOCAL in dirty
        judge = got
        if scope:
            heads = {d: head_text(repo_disk, d) for d in scope}

            def _read(p: Path, _heads=heads, _repo=repo_disk) -> str | None:
                rel = p.name if p.parent == _repo else str(p.relative_to(_repo))
                if rel in _heads:
                    return _heads[rel]      # None = not in HEAD (a NEW file)
                return _worktree_read(p)

            judge = audit(repo, sibs, alloc, docs, crates, patterns, read=_read)
            wt_keys = {_row_key(r) for c in _CLASSES for r in got[c]}
            hd_keys = {_row_key(r) for c in _CLASSES for r in judge[c]}
            n_uncommitted += len(wt_keys - hd_keys)
            masked = [r for c in _CLASSES for r in judge[c]
                      if _row_key(r) not in wt_keys]
            n_masked += len(masked)
            for r in masked:
                print(f"⛔ MASKED  {repo.name}: {r}")
        if scope or hw_dirty:
            dirty_scope[repo.name] = len(scope) + int(hw_dirty)

        row = pins.get(repo.name)
        tot["docs"] += got["n_docs"]
        tot["cites"] += got["n_cites"]
        tot["amb"] += len(got["ambiguous"])
        tot["mis"] += got["misleading"]
        tot["misat"] += got["misattributed"]
        tot["rep"] += got["repeat"]
        tot["width"] += got["unseen_width"]
        tot["trail"] += got["alias_trailing"]
        units = len(got["cross_units"])
        # SUM, never union: riir-auth and riir-game-sdk both citing Plan 488
        # is TWO adjudications in two documents, not one. A union reported 142
        # where the work is 164.
        tot["units"] += units
        # ORACLE_STALE rides this loop for the TOTAL only. It is deliberately
        # absent from every pin comparison: a ceiling that can be breached by
        # another repo's fetch schedule is the cries-wolf failure Issue 827
        # exists to stop, and a ratchet on an UNDECIDED bucket is a backlog
        # wearing a pin (Issue 785's rule).
        for cls in (CROSS, IN_RANGE, ORPHAN, MISATTR_IN_RANGE, ORACLE_STALE):
            tot[cls] += len(got[cls])
        if repo_disk.resolve() == REPO_ROOT:
            mine_row = got

        flags = []
        if row is None:
            # Issue 821: an acknowledged known-extra owes no pin row —
            # the marker reached population_verdict's FINAL line and not
            # this loop, so 8 of 9 sweeps red on repos they found
            # nothing in, hiding two live ratchet breaches.
            if not pin_row_exempt(repo.name):
                flags.append("UNPINNED — add a row (or it can never red)")
        else:
            # `judge`, not `got` — the pins adjudicate HEAD (Issue 797). On a
            # clean repo the two ARE the same object, so this is a no-op in the
            # ordinary case and the distinction costs nothing.
            if judge["n_cites"] < row["min_citations"]:
                flags.append(f"walk FLOOR breached: {judge['n_cites']} citations < "
                             f"{row['min_citations']} — prose was removed, or the "
                             f"citation regex went blind (which reads as clean)")
            for cls, key in ((CROSS, "max_cross"), (IN_RANGE, "max_in_local_range"),
                             (ORPHAN, "max_orphan")):
                if len(judge[cls]) > row[key]:
                    flags.append(f"{cls} {len(judge[cls])} > pinned {row[key]}")
        # Issue 794 — a GLOBAL wall, deliberately not a per-repo ratchet field:
        # the class has no backlog anywhere (1 row workspace-wide at landing,
        # repaired in the same commit), so per-repo pins would be 16 zeros and
        # a 5th field on every row for a quantity that is 0 by contract.
        if len(judge[MISATTR_IN_RANGE]) > glob_wall:
            flags.append(f"{MISATTR_IN_RANGE} {len(judge[MISATTR_IN_RANGE])} > "
                         f"pinned {glob_wall} — a citation that is FOLLOWABLE "
                         f"to the WRONG repo; the local-range excuse does not "
                         f"apply (this repo never allocated the number)")
        findings = got[CROSS] + got[IN_RANGE] + got[ORPHAN]
        status = "✗" if flags else ("·" if findings else "✓")
        # `cross` counts EDITS, `over N num` counts ADJUDICATIONS — and they are
        # not the same job. Inserting the repo name is mechanical; deciding
        # WHICH owner a sentence means (most numbers have several) is the
        # expensive part, and it is paid once per number, not once per row.
        # Measured at landing the ratio runs 1.0x to 4.2x, so a repo owner
        # sizing the work from the row count alone is wrong by up to 4x:
        # riir-viewbridge's 17 rows are FOUR decisions. Same standing as tail
        # support in the percentile audit — it ORDERS the work, it is not a
        # second verdict, and neither number is the finding count on its own.
        acc, shp, nov = blind[repo.name]
        # Issue 921 Arm B: the derived two-lane floor, printed so the
        # mechanism is visible on every run for the repos that DECLARE a
        # local lane. Empty string — zero bytes — without the file.
        gap_s = "".join(f" {k.lower()}_gap={c}..{'open' if f is None else f}"
                        for k, (c, f) in sorted(got["gap"].items()))
        # `novel` is the part of `heading_unread` that could change ANY
        # verdict (Issue 828 T4): the rest is already known from a file. Shown
        # beside the count, never instead of it - the count is what says
        # whether this repo's IN-LOCAL-RANGE figure can be read as an
        # editorial quantity at all.
        style = (f" heading_unread={shp - acc}/{shp} novel={nov}") if shp else ""
        print(f"{status} {repo.name:22s} docs={got['n_docs']} cites={got['n_cites']:<5d} "
              f"cross={len(got[CROSS]):<4d} over {units:<3d} num "
              f"in_local_range={len(got[IN_RANGE]):<3d} "
              f"orphan={len(got[ORPHAN])} ambiguous={len(got['ambiguous'])}"
              f"{gap_s}{style}")
        # 12 rows keeps the whole-workspace run readable; `--full` is for the
        # one job the truncated view cannot do — writing the OWNING repo's
        # issue, which needs every row it is being asked to repair.
        cap = len(got[CROSS]) if FULL else 12
        for r in got[CROSS][:cap]:
            print(f"      cross:    {r}")
        if len(got[CROSS]) > cap:
            print(f"      … {len(got[CROSS]) - cap} more cross row(s) "
                  f"(re-run with --full)")
        # ahead of the undecided list, and NEVER truncated: these are findings.
        for r in got[MISATTR_IN_RANGE]:
            print(f"      ⛔wrong-addr:{r}")
        for r in got[IN_RANGE][:(len(got[IN_RANGE]) if FULL else 4)]:
            print(f"      undecided:{r}")
        for r in got[ORPHAN]:
            print(f"      orphan:   {r}")
        # Issue 827. Never truncated and never counted: a row here is a
        # statement about OUR checkout of somebody else's repo, and the reader
        # needs the whole list to know which fetch would settle it.
        for r in got[ORACLE_STALE]:
            print(f"      ⚠oracle:  {r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    # The population axis, shared (Issue 793): UNREGISTERED reds in every
    # posture, UNSEEN reds without the marker, and the same set DEFERS loudly
    # with it. Never auto-detected — a genuine removal whose row update was
    # forgotten is set-identical to a partial clone from the walk alone.
    pop_lines, deferred, pop_fail = population_verdict(pins, {r.name for r in repos})
    for _line in pop_lines:
        print(_line)
    if pop_fail:
        bad = True

    # Issue 797. Rides the FINAL line in BOTH directions, the `deferred`
    # precedent — a notice printed only on failure is one nobody reads on the
    # run that passes. ADVISORY and not a failure: a sweep that hard-reds on an
    # ordinary dirty worktree is a sweep nobody runs. MASKED is the exception
    # and it already reds through the pins, because `judge` counts it.
    # ⛔ The UPSTREAM axis, which this sweep did not have. It calls the
    # low-level `worktree_advisory()` rather than `sweep_advisory()` — for a
    # good reason, its scope is `dirty_files` intersected with its OWN
    # document set, which is sharper than any glob — and the cost was that it
    # got the worktree axis and NOTHING about being behind origin, in the one
    # sweep whose rows carry a `file:line` address and name another repo.
    #
    # Measured: it reported a CROSS finding at riir-neuron-db `HISTORY.md:49`
    # with no advisory at all, while `origin/develop` already carried the
    # repair and that checkout was 4 commits behind with one of them touching
    # that very file. Issue 798's founding class, *a committed FIX read
    # dirty*, and it cost an investigation before `git show origin/develop`
    # settled it.
    #
    # `upstream_axis` is the shared half of `sweep_advisory`, so the two entry
    # points now mean the same thing — which is what
    # `sweep_advisory_membership_gate` has been asserting all along by
    # accepting either. The patterns are the sweep's OWN documents, not a
    # glob: `behind_origin` should be asked about exactly the files whose
    # staleness could move a row — plus `.highwater_local` (Issue 921 Arm B),
    # which moves the two-lane gap and so can move a verdict from upstream.
    _stale, _unver = upstream_axis([repo_alias.real(r) for r in repos],
                                   tuple(docs) + (HW_LOCAL,))
    deferred.extend(worktree_advisory(dirty_scope, n_uncommitted, n_masked,
                                      stale=_stale, unverified=_unver))

    # ── T4, second half: the katgpt-rs row must EQUAL the gate's own run ─────
    rc, scanned, findings = gate_says()
    if scanned == -2:
        # The gate DEFERRED its cross-repo adjudication on this box. Its own
        # numbers are not an adjudication, so there is nothing to cross-check
        # against — recorded as a deferral, never as an agreement and never as
        # a broken instrument.
        deferred.append(f"the {GATE.name} cross-check — the gate DEFERRED its "
                        "cross-repo adjudication on this partial clone, so "
                        "there is no number to assert this sweep's row against")
    elif rc == 2 or scanned < 0:
        print(f"✗ INSTRUMENT: {GATE.name} itself reported untrustworthy (rc={rc}) "
              f"— its numbers cannot cross-check this sweep's")
        return 2
    else:
        mine_total = sum(len(mine_row[c]) for c in (CROSS, IN_RANGE, ORPHAN))
        if (scanned, findings) != (mine_row["n_cites"], mine_total):
            bad = True
            print(f"✗ CROSS-CHECK: {GATE.name} scanned {scanned} / found "
                  f"{findings}; this sweep's {REPO_ROOT.name} row is "
                  f"{mine_row['n_cites']} / {mine_total}. Same documents, same "
                  f"regex — they cannot disagree. (The sweep PARTITIONS the "
                  f"gate's finding set into {CROSS}/{IN_RANGE}/{ORPHAN}; the "
                  f"total must match.)")
        else:
            print(f"\n  cross-check vs {GATE.name}: {scanned} citations / "
                  f"{findings} finding(s) — AGREE (asserted, not assumed)")

    print(f"{len(repos)} contract repo(s) · {tot['docs']} document(s) · "
          f"{tot['cites']} citation(s) · {tot[CROSS]} CROSS over "
          f"{tot['units']} per-repo adjudication(s) · "
          f"{tot[IN_RANGE]} IN-LOCAL-RANGE · {tot[ORPHAN]} ORPHAN · "
          f"{tot[ORACLE_STALE]} ORACLE-STALE")
    if tot[ORACLE_STALE]:
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(unreliable.items()))
        print(f"  ⚠ ORACLE-STALE is UNDECIDED, never clean and never counted "
              f"(Issue 827): {tot[ORACLE_STALE]} row(s) name a repo whose "
              f"checkout here cannot answer whether it owns the number — "
              f"{detail}. Measured once already as four ⛔MISATTRIBUTED rows "
              f"and a breached ceiling in riir-shader, over citations that "
              f"were CORRECT. `git fetch` in the named repo and re-run.")
    print(f"  AMBIGUOUS (local AND sibling — undecidable by number, NOT a pass): "
          f"{tot['amb']}  ·  ⛔MISLEADING crate hints: {tot['mis']}"
          f"  ·  ⛔MISATTRIBUTED (names a NON-owner repo): {tot['misat']}"
          f"  ·  ⛔MISATTRIBUTED-IN-RANGE (Issue 794 — FOLLOWABLE to the wrong "
          f"repo, walled at {glob_wall}): {tot[MISATTR_IN_RANGE]}")
    b_acc = sum(a for a, _, _ in blind.values())
    b_shp = sum(t for _, t, _ in blind.values())
    b_nov = sum(n for _, _, n in blind.values())
    # Issue 823 T6. The meter's own blindness detector. Its failure mode is a
    # PERFECT-LOOKING score: a regressed shaped pattern takes `b_shp` to 0 and
    # the line below reads `0/0 records read, 0 UNREAD`. GLOBAL, because
    # per-repo is legitimately 0 wherever a repo has no self-allocation
    # headings. Deliberately NOT gated on the partial-clone marker: this floor
    # asserts the PARSER, not the population, and it is pinned far enough under
    # a partial-clone measurement that an absent repo cannot breach it.
    if b_shp < glob["min_heading_shaped"]:
        bad = True
        print(f"✗ heading WIDTH BOUND breached: {b_shp} heading-shaped "
              f"self-allocation record(s) < pinned "
              f"{glob['min_heading_shaped']} — the meter that prints the "
              f"oracle's blindness has itself gone blind, and its output in "
              f"that state ({b_acc}/{b_shp}) reads as PERFECT COVERAGE. This "
              f"is a PARSE regression, not a population change.")
    # Issue 828 T4 - the COST of the unread, which decides Issue 823 T5's
    # open question and had never been taken. Printed every run because it
    # moves whenever a sibling edits a heading, and because a bare UNREAD
    # count reads as a backlog while its price reads as a rounding error.
    print(f"  heading oracle COST (Issue 828 T4): of the {b_shp - b_acc} "
          f"UNREAD, {b_nov} contribute a number NO other oracle knows "
          f"(worktree file or `git log`). Only those can change a verdict, "
          f"so that is the entire blast radius of the `## Issue NNN resolved "
          f"— title (date)` family Issue 823 T5 left open. The rule stays "
          f"UNSOUND to widen (arm 2 pins `follow-up` as a negative, and no "
          f"punctuation rule separates commentary from allocation) — and now "
          f"it is also not worth widening. Take both figures from THIS line.")
    print(f"  heading oracle (Issue 781): {b_acc}/{b_shp} self-allocation "
          f"records read, {b_shp - b_acc} UNREAD **on style alone** — a "
          f"triage quantity, never a verdict. `heading_allocated()` requires "
          f"the number to be followed IMMEDIATELY by its delimiter, so "
          f"`## Issue 042 (date) — title`, (Issue 823) the date-led "
          f"`## <date> — Issue 113: title` and (Issue 828) the "
          f"dash-delimited `## Issue 788 — title` all read, while `## Issue 097 "
          f"resolved — title (date)` does not: the split is by HOUSE STYLE "
          f"rather than correctness. Dropping that DISCRIMINATOR is still "
          f"UNSOUND and selftest arm 2 proves it — `## Issue 043 follow-up "
          f"(date)` is a pinned negative, in both positions. Adding a "
          f"POSITION is not that widening; Issue 823 measured 74 date-led "
          f"records the meter itself could not see, two repos printing a "
          f"PERFECT score over 20+ unread. The residual cost lands as "
          f"UNDECIDED noise on the local side and as a FALSE ⛔MISATTRIBUTED "
          f"on the owners side.")
    print(f"  of the {tot[CROSS]} CROSS: {tot['rep']} carry the REPEAT label — "
          f"the same document already attributes that number elsewhere, so the "
          f"repair is mechanical (copy it), not a lookup. The labels are "
          f"mutually exclusive and ⛔MISATTRIBUTED outranks REPEAT, so the "
          f"mechanically-repairable population is {tot['rep']}+ , not exactly "
          f"{tot['rep']}")
    # TWO error rates over TWO populations, never blended into one number: a
    # SAMPLE rate does not transfer to rows it never sampled (Issue 752).
    print(f"  ⛔ measured FALSE-POSITIVE rates, by population — do NOT quote "
          f"{tot[CROSS]} without them, nor without the {tot['cites']}-citation "
          f"walk and {len(repos)}-repo population that produced them:")
    print(f"       254 rows (the pre-752 corpus): 7/43 = 16%, a STRATIFIED "
          f"SAMPLE read across 13 repos (Issue 751 T1)")
    print(f"       +45 rows recovered by owner-consistency: 1/45, a full CENSUS "
          f"— every row read (Issue 752), ONE refuted afterwards by a "
          f"measurement the census could not make (Issue 754: a census "
          f"inherits its oracle's blind spots at 100%). {tot['misat']} in "
          f"CROSS carry an explicit non-owner attribution; hand-adjudicated, "
          f"2 are outright WRONG addresses (`riir-chain Plan 211` — riir-chain "
          f"tops out at 058), the rest unqualified either way")
    # ── the two RULE-COST quantities (Issue 753) ────────────────────────────
    # Both were docstring claims measured once; both are re-measured every run
    # now, because the corpus moves and a dated zero is a claim, not a fact.
    print(f"  width bound `\\d{{2,4}}`: {tot['width']} single-digit form(s) in "
          f"scope — NOT scanned, by design. Widening to `\\d{{1,4}}` was measured "
          f"over every tracked .md in the workspace at 51 FALSE heads "
          f"(`## Bench 1:` section numbering) and 0 true ones, so the bound "
          f"stays and the class is WATCHED (pinned 0 in issue_citation_floors.txt)")
    print(f"  alias reach is LEAD-only: {tot['trail']} of the {tot[CROSS]} CROSS "
          f"rows would be SUPPRESSED by a forward-reaching alias. Read line by "
          f"line at landing, 0 of them were genuine attributions (the one row is "
          f"`chain` inside prose about the `chain_viz` crate), so widening buys 0 "
          f"repairs and hides true findings — the backward-only window's argument "
          f"on a second axis. A triage quantity, never a verdict")
    print(f"  scope: AGENTS.md + HISTORY.md only ({'/'.join(docs)}, pinned in "
          f"{GATE_PINS.name}). A walk of *.md would pull in .plans/.docs/"
          f".research — thousands of by-design LOCAL citations — and drown the "
          f"signal. A `cross=0` above is NOT a claim about those.")
    print(f"  IN-LOCAL-RANGE is UNDECIDED, never 'clean': the number is at or "
          f"under the repo's own top allocation, so a local referent that was "
          f"skipped or never committed is plausible. EXCEPT the two-lane gap "
          f"(Issue 921 Arm B): a repo declaring `.highwater_local` classifies "
          f"counter < n < floor as CROSS/ORPHAN — the range above its local "
          f"lane is inherited, not local-intent.")

    if bad:
        print("✗ citation sweep FAILED — see the ✗ rows above")
        for _d in deferred:
            print(f"  {deferral_line(_d)}")
        print("    Fix: name the owning repo in the prose — `riir-ai Issue 750`, "
              "`riir-train Issue 513`. The number alone is not an address.")
        return 1
    _line = "✓ citation sweep PASSED — every repo at or under its pinned ratchet"
    if deferred:
        _line += "; DEFERRED: " + "; ".join(deferred)
    print(_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
