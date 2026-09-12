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
`"seal-remake" in ctx` read riir-viewbridge's `seal-remake-unity` as naming
**seal-remake** and qualified a `Plan 031` citation on a different repo's name.

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

Two audited sub-questions (Issue 751 T2) — both decided with evidence
--------------------------------------------------------------------
**A crate name is NOT a qualification form.** The workspace's crate->repo map
is unique (156 package names, ZERO collisions across 19 repos), so
`riir-games-mmorpg::…` *could* address riir-ai mechanically — but the map
lives in no document the reader has, and the measurement refutes the weaker
claim too: of the 45 CROSS rows carrying a crate name in the window, **4
resolve to a repo that does NOT own the cited number** (riir-game-sdk's
`Issue 097` sits next to `riir-games-mmorpg::sync_facades::avatar_sync` while
097 belongs to riir-mmorpg-examples/riir-chain/seal-game-editor). Accepting
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

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "citation_drift_floors.txt"
GATE_PINS = HERE / "issue_citation_floors.txt"
GATE = HERE / "issue_citation_gate.py"

FIELDS = ("min_citations", "max_cross", "max_in_local_range", "max_orphan")

CROSS, IN_RANGE, ORPHAN = "CROSS", "IN-LOCAL-RANGE", "ORPHAN"

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
        ls = subprocess.run(["git", "-C", str(r), "ls-files", "*Cargo.toml"],
                            capture_output=True, text=True)
        for rel in ls.stdout.split():
            try:
                text = (r / rel).read_text(encoding="utf-8", errors="replace")
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


def audit(repo: Path, sibs: list[Path], alloc: dict[str, dict[str, set[int]]],
          docs: list[str], crates: dict[str, str],
          patterns: dict[str, re.Pattern]) -> dict:
    """One repo -> the three finding classes + BOTH populations under them."""
    mine = alloc[repo.name]
    top = top_allocated(repo, mine)
    elsewhere: dict[str, dict[int, list[str]]] = {k: {} for k in icg.KINDS}
    for s in sibs:
        for kind in icg.KINDS:
            for n in alloc[s.name][kind]:
                elsewhere[kind].setdefault(n, []).append(s.name)

    got = {"n_docs": 0, "n_cites": 0, "ambiguous": set(), "misleading": 0,
           "misattributed": 0, "cross_units": set(), "repeat": 0,
           "unseen_width": 0, "alias_trailing": 0,
           CROSS: [], IN_RANGE: [], ORPHAN: []}
    for doc in docs:
        p = repo / doc
        if not p.is_file():
            continue          # not every repo carries a HISTORY.md — absence
        got["n_docs"] += 1    # is not a finding, but the doc COUNT is printed
        text = p.read_text(encoding="utf-8", errors="replace")
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
            ctx = "\n".join(lines[max(0, ln - 3):ln])
            hint = _crate_hits(ctx, crates, patterns) - {repo.name}
            cls = (IN_RANGE if n <= top[kind] else CROSS if owners else ORPHAN)
            tag = ""
            bad = adj - set(owners)
            if cls is CROSS and bad:
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
                f"{f' (local top {top[kind]})' if cls is IN_RANGE else ''}{tag}\n"
                f"          {lines[ln - 1].strip()[:110]}")
    return got


def gate_says() -> tuple[int, int, int]:
    """Run the per-push gate and READ its numbers. The sweep re-states a
    quantity the gate owns; asserting beats trusting. -> (rc, scanned, findings)"""
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True)
    scanned = re.search(r"scanned (\d+) citations", r.stdout)
    failed = re.search(r"FAILED — (\d+) unqualified", r.stdout)
    return (r.returncode,
            int(scanned.group(1)) if scanned else -1,
            int(failed.group(1)) if failed else (0 if r.returncode == 0 else -1))


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
        (me / ".issues" / "010_local.md").write_text("x")
        for n in ("010", "500", "600"):
            (sib / ".issues" / f"{n}_sib.md").write_text("x")
        (me / "crates" / "thing").mkdir(parents=True)
        (me / "crates" / "thing" / "Cargo.toml").write_text('[package]\nname = "sibcrate-x"\n')
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
            "`sibcrate-x` ships it; Issue 600 rides the crate.\n")  # crate-hint
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
        (me / "AGENTS.md").write_text("riir-fakesib Issue 500 is qualified.\n")
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
        (me / "AGENTS.md").write_text("`sibcrate-x` ships it; Issue 500 rides the crate.\n")
        hint = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2)
        if len(hint[CROSS]) != 1 or "crate-hint" not in hint[CROSS][0]:
            fails.append(f"crate-hint sub-class did not fire: {hint[CROSS]}")
        if hint["misleading"]:
            fails.append("crate naming the OWNER was counted as misleading")

        (me / "AGENTS.md").write_text("`othercrate-y` moved it; Issue 500 is elsewhere.\n")
        mis = audit(me, [sib, other], alloc, ["AGENTS.md"], crates2, pats2)
        if mis["misleading"] != 1 or len(mis[CROSS]) != 1:
            fails.append(f"MISLEADING sub-class did not fire: {mis['misleading']} "
                         f"{mis[CROSS]}")

        # ── the two RULE-COST probes must FIRE, and their controls must NOT
        # (Issue 753). Both report a 0 in the live workspace, which is exactly
        # the shape a probe wired to nothing also reports.
        (me / "AGENTS.md").write_text(
            "## Bench 1: a section heading, not a citation\n"
            "Issue 500 is the real one.\n")
        w = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if w["unseen_width"] != 1:
            fails.append(f"width complement did not fire: {w['unseen_width']} != 1")
        if w["n_cites"] != 1:
            fails.append(f"width: `Bench 1` must NOT enter the walk, got "
                         f"{w['n_cites']} citations")
        (me / "AGENTS.md").write_text("Issue 500 alone.\n")
        if audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)["unseen_width"]:
            fails.append("width complement counted a 3-digit citation")

        # an alias AFTER the citation: still CROSS (the rule is lead-only), and
        # counted as the cost of that decision.
        (me / "AGENTS.md").write_text("Issue 500, over in fakesib somewhere.\n")
        tr = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if len(tr[CROSS]) != 1:
            fails.append(f"a TRAILING alias must not qualify: {tr[CROSS]}")
        if tr["alias_trailing"] != 1:
            fails.append(f"alias-trailing cost did not fire: {tr['alias_trailing']}")
        # CONTROL A: the same alias in the LEAD qualifies, so no row and no cost.
        (me / "AGENTS.md").write_text("fakesib Issue 500 is addressed.\n")
        lead = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if lead[CROSS] or lead["alias_trailing"]:
            fails.append(f"lead alias must qualify: {lead[CROSS]} {lead['alias_trailing']}")
        # CONTROL B: a trailing alias of a NON-owner is not a suppression cost.
        (me / "AGENTS.md").write_text("Issue 500, over in otherlib somewhere.\n")
        if audit(me, [sib, other], alloc, ["AGENTS.md"], crates, pats)["alias_trailing"]:
            fails.append("alias-trailing counted a NON-owner alias")

        # padding is the SAME number (Issue 751 T2b) — `006` must read as 6
        (me / "AGENTS.md").write_text("Issue 0500 no; Issue 500 yes.\n")
        pad = audit(me, [sib], alloc, ["AGENTS.md"], crates, pats)
        if len(pad[CROSS]) != 2:
            fails.append(f"zero-padding: `Issue 0500` and `Issue 500` must be "
                         f"ONE number, got {len(pad[CROSS])} CROSS rows")

        # pin parser: globals + 5-field rows, comments stripped, arity enforced
        pins = ws / "pins.txt"
        pins.write_text("# c\nmin_repos = 15\nrepo-a 10 0 0 0  # trailing\n\n")
        g, rows = parse_pins(pins)
        if g != {"min_repos": 15} or rows != {"repo-a": dict(zip(FIELDS, (10, 0, 0, 0)))}:
            fails.append(f"pin parse: got {g} {rows}")
        pins.write_text("repo-a 1 2\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass

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
            "## Plan 046 (2026-01-01) — a different KIND\n")
        names = ["riir-headrepo", "riir-fakesib"]
        got_h = icg.heading_allocated(hd, ".issues", names)
        if got_h != {42}:
            fails.append(f"heading allocation: got {sorted(got_h)}, expected [42] "
                         f"— 43/44/45 are the measured negatives, 46 is a Plan")
        if icg.heading_allocated(hd, ".plans", names) != {46}:
            fails.append("heading allocation: the KIND is not read from the subdir")
        if icg.allocated(hd, ".issues", names) != {42}:
            fails.append("allocated() does not union the heading path")

        # population derivation: BOUNDARY.md + a .git DIRECTORY, both required
        (me / "BOUNDARY.md").write_text("x")
        (me / ".git").mkdir()
        (sib / "BOUNDARY.md").write_text("x")
        (sib / ".git").write_text("gitdir: elsewhere")   # worktree-shaped
        if [p.name for p in icg.contract_repos(ws)] != ["fake-repo"]:
            fails.append(f"population derivation wrong: "
                         f"{[p.name for p in icg.contract_repos(ws)]}")
    return fails


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

    crates = crate_map(repos)
    patterns = {c: re.compile(r"\b" + re.escape(c).replace(r"\-", "[-_]") + r"\b")
                for c in crates}
    alloc = {r.name: {k: icg.allocated(r, d) for k, d in icg.KINDS.items()}
             for r in repos}

    bad = False
    tot = {"docs": 0, "cites": 0, "amb": 0, "mis": 0, "misat": 0,
           "units": 0, "rep": 0, "width": 0, "trail": 0,
           CROSS: 0, IN_RANGE: 0, ORPHAN: 0}
    mine_row = None
    for repo in repos:
        sibs = [s for s in repos if s != repo]
        got = audit(repo, sibs, alloc, docs, crates, patterns)
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
        for cls in (CROSS, IN_RANGE, ORPHAN):
            tot[cls] += len(got[cls])
        if repo.resolve() == REPO_ROOT:
            mine_row = got

        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_cites"] < row["min_citations"]:
                flags.append(f"walk FLOOR breached: {got['n_cites']} citations < "
                             f"{row['min_citations']} — prose was removed, or the "
                             f"citation regex went blind (which reads as clean)")
            for cls, key in ((CROSS, "max_cross"), (IN_RANGE, "max_in_local_range"),
                             (ORPHAN, "max_orphan")):
                if len(got[cls]) > row[key]:
                    flags.append(f"{cls} {len(got[cls])} > pinned {row[key]}")
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
        print(f"{status} {repo.name:22s} docs={got['n_docs']} cites={got['n_cites']:<5d} "
              f"cross={len(got[CROSS]):<4d} over {units:<3d} num "
              f"in_local_range={len(got[IN_RANGE]):<3d} "
              f"orphan={len(got[ORPHAN])} ambiguous={len(got['ambiguous'])}")
        # 12 rows keeps the whole-workspace run readable; `--full` is for the
        # one job the truncated view cannot do — writing the OWNING repo's
        # issue, which needs every row it is being asked to repair.
        cap = len(got[CROSS]) if FULL else 12
        for r in got[CROSS][:cap]:
            print(f"      cross:    {r}")
        if len(got[CROSS]) > cap:
            print(f"      … {len(got[CROSS]) - cap} more cross row(s) "
                  f"(re-run with --full)")
        for r in got[IN_RANGE][:(len(got[IN_RANGE]) if FULL else 4)]:
            print(f"      undecided:{r}")
        for r in got[ORPHAN]:
            print(f"      orphan:   {r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - {r.name for r in repos}):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    # ── T4, second half: the katgpt-rs row must EQUAL the gate's own run ─────
    rc, scanned, findings = gate_says()
    if rc == 2 or scanned < 0:
        print(f"✗ INSTRUMENT: {GATE.name} itself reported untrustworthy (rc={rc}) "
              f"— its numbers cannot cross-check this sweep's")
        return 2
    mine_total = sum(len(mine_row[c]) for c in (CROSS, IN_RANGE, ORPHAN))
    if (scanned, findings) != (mine_row["n_cites"], mine_total):
        bad = True
        print(f"✗ CROSS-CHECK: {GATE.name} scanned {scanned} / found {findings}; "
              f"this sweep's {REPO_ROOT.name} row is {mine_row['n_cites']} / "
              f"{mine_total}. Same documents, same regex — they cannot disagree. "
              f"(The sweep PARTITIONS the gate's finding set into "
              f"{CROSS}/{IN_RANGE}/{ORPHAN}; the total must match.)")
    else:
        print(f"\n  cross-check vs {GATE.name}: {scanned} citations / {findings} "
              f"finding(s) — AGREE (asserted, not assumed)")

    print(f"{len(repos)} contract repo(s) · {tot['docs']} document(s) · "
          f"{tot['cites']} citation(s) · {tot[CROSS]} CROSS over "
          f"{tot['units']} per-repo adjudication(s) · "
          f"{tot[IN_RANGE]} IN-LOCAL-RANGE · {tot[ORPHAN]} ORPHAN")
    print(f"  AMBIGUOUS (local AND sibling — undecidable by number, NOT a pass): "
          f"{tot['amb']}  ·  ⛔MISLEADING crate hints: {tot['mis']}"
          f"  ·  ⛔MISATTRIBUTED (names a NON-owner repo): {tot['misat']}")
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
          f"skipped or never committed is plausible.")

    if bad:
        print("✗ citation sweep FAILED — see the ✗ rows above")
        print("    Fix: name the owning repo in the prose — `riir-ai Issue 750`, "
              "`riir-train Issue 513`. The number alone is not an address.")
        return 1
    print("✓ citation sweep PASSED — every repo at or under its pinned ratchet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
