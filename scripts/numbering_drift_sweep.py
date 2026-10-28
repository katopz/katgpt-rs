#!/usr/bin/env python3
"""Run the numbering gate's checks over EVERY contract repo, not just this one.

`scripts/numbering_gate.py` is katgpt-rs-scoped by construction: its pins live
in `scripts/numbering_floors.txt`, which says "katgpt-rs scope only" in its
first line, and `docs_gate.yml` has a single checkout so it could never see a
sibling. That is the right shape for a per-push CI gate and the wrong shape for
the question "is anyone else allocating numbers twice?" — which nothing had
ever asked. Measured the first time it was asked (2026-09-05):

    32 tracked duplicate numbers across 4 sibling repos, in ALLOCATOR-SERIAL
    directories, while katgpt-rs — the one repo with a gate — had zero.
     5 `.highwater` files that are not integers at all, every one of them
       `echo -n <N> > .highwater` writing its own flag into the file, which
       DISARMS the above-highwater check for that directory (Issue 725).

This is the same shape as Issue 702 one instrument over: an auditor that
accepts a repo path, pointed at exactly one repo for months, so a sibling with
the defect is indistinguishable from a sibling nobody looked at.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical reasoning to scripts/docs_drift_sweep.py: CI has one checkout, the
siblings are private and simply absent, so this would either red on every run
or derive an empty population and print a confident green over zero repos.

    this script              workstation, on demand, every contract repo
    numbering_gate.py        CI, per-push (docs_gate.sh), katgpt-rs only

Vocabulary vs population
------------------------
The population (which repos exist) is DERIVED — BOUNDARY.md + a `.git` dir,
never typed, per Issue 703. The expectations are COMMITTED, in
`scripts/numbering_drift_floors.txt`, because deriving both from one walk is
what makes a cross-repo gate permanently green.

Ceilings here are a RATCHET, not a wall — with ONE exception: the classified
collision class walls at zero in every repo, because a filename-decidable
two-owner number is not an arbitration backlog, it is a defect (riir-train
issue-556's law: the pinned COLLISION row is a work item, never an excuse).
Everything else keeps the measured-ratchet rule: a new collision of the
ratcheted classes reds immediately, and the standing backlog is visible in the
pins rather than silently tolerated. Lower a pin in the commit that resolves a
collision.

Scope, which is the load-bearing decision
-----------------------------------------
Raw `dup` + above-highwater checks run ONLY over allocator-serial directories
(`.plans/.issues/.research/.proposals`). `.benchmarks/` and `.docs/` are
excluded from the RAW count for exactly the reason numbering_floors.txt gives:
there the leading number is the OWNING plan/issue and a family per owner is
the intended convention, so "duplicate" is not decidable without knowing which
convention a given file follows — and a gate must not guess.

⛔ But the classifier (numbering_gate.classify_stems, ported from riir-train
issue-556's gate 2026-10-29) RETIRED the decidability excuse for the classes
that ARE decidable from the stems. Since then every ≥2-tracked-holder group in
ALL six directories classifies, and the classified classes gate over every
directory:

  collision       token-disjoint holders (two unrelated owners on one number)
                  — a WALL at zero, never pinned, never a ratchet. The fix is
                  the issue-556 renumber protocol (loser moves, every citation
                  moves in the same commit). Found live the day it landed:
                  riir-ai 809/872/1008 + riir-infer 884, all four resolved
                  before the wall armed (2d591ea, b5f9a8c2f, 4427a6547).
  rename_residue  one document under two spellings — ratchet at measured (0
                  everywhere at landing; raise only with the adjudication in
                  the commit message).
  hand_read       undecidable from filenames (no owner tokens — the
                  old-convention multi-record bench numbers). RATCHET at
                  measured: the standing backlog stays visible in the pins, a
                  NEW token-less group reds. The pairwise token rule still
                  catches a token-carrying intruder inside a hand-read family.
  FAMILY-EXPLICIT the intended convention — counted, disclosed, never a
                  finding (the `fam=` column).

The population floor `min_bench` (tracked numbered documents in
`.benchmarks/` + `.docs/`) sits under the new classes the same way `min_files`
sits under the serial ones: a wall is only as good as the walk beneath it.

The rename/hand CAPS for THIS repo are additionally SYNCED on every run
against the per-push gate's scalars (`numbering_floors.txt`, the CI half
landed 2026-10-29 in `numbering_gate.py`): two instruments, one classifier
(imported), so pin divergence between the two floors files is the only way
their verdicts can disagree — `classified_pin_sync` refuses loud (exit 2)
before any repo verdict. Floors are deliberately not synced (different slack
under the same measured population is not a record of the same quantity).

The MALFORMED check runs over every numbered directory including those two,
because a file that is not an integer is undecidable under no convention.

Report + gate. Exit 0 clean, 1 on drift above the pins, **2 if the instrument
itself is untrustworthy** (selftest failure) — an unreliable instrument is not
the same finding as drift and must not be reported as one.

The counter-transition classes (Issues 769 + 770, landed 2026-09-13)
---------------------------------------------------------------------
`.highwater` is not only checked for CONTENT (malformed / above-max) but for
HISTORY: `highwater_contiguity_audit.py`'s walker classifies every commit
that touched the counter against ITS OWN PARENTS' blob values (the Issue-770
repair — the 769 walker ordered all refs' hunks by commit DATE, which
manufactured phantom resets from lineage interleaving and could not see
MERGE commits at all; `git log -p` emits no merge diffs). Two classes are
verdicts here —

  resets     a commit (or merge resolution) whose counter value is below
             max(parent values). After `517→511` the re-climb re-spends
             numbers — the never-reuse rule broken by construction, the
             `.issues/121` collision class at counter granularity. Each row
             names the commit that DID it (merge commits included — a merge
             taking the lower side IS the hazard). First corrected
             adjudication (Issue 770): 27 resets workspace-wide, a mix of
             stale-lineage writebacks and merge resolutions taking the lower
             side; historical hazard absorbed by gap fast-forwards and
             push-wins renumbers; the max_dup column proves no live file
             duplicates. Pinned at the MEASURED count per repo — a NEW reset
             reds immediately, the standing history stays visible in the pins.
  unbumped   the WORKTREE counter sits below its committed history max on
             HEAD — a checkout/branch state, not a commit. REPORT-ONLY,
             deliberately unpinned (Issue 770 finding 4): the quantity is a
             function of which branch the box carries (mmorpg-editor's
             bevy worktree vs other refs' 194), so a pin would be ref-set
             dependent and red on a box that owes nothing — the class the
             AGENTS.md "a verdict the box can invalidate should refuse" law
             assigns to printing, never gating.

Both run over EVERY dir carrying a `.highwater`, not just SERIAL_DIRS — a
backward move is anomalous under number- and count-based conventions alike
(the malformed check's precedent). HEAD-reachable only, never --all: pins
must be a function of the branch the box carries, not of which refs happen
to be fetched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numbering_gate as ng  # noqa: E402  (DRY: one scanner, two cadences)
import highwater_contiguity_audit as hca  # noqa: E402  (DRY: one transition walker, two cadences — Issue 769)
from sweep_population import open_repo, population_verdict, pin_row_exempt  # noqa: E402
from worktree_state import (  # noqa: E402
    HeadDelta, deferral_line, delta_of, head_text, sweep_advisory)
import repo_alias  # noqa: E402 — the machine-local name codec (see its docstring)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
PINS = Path(__file__).resolve().parent / "numbering_drift_floors.txt"
# The per-push gate's pins — the classified-class CAPS below are ONE measured
# record in two media across these two files, and the sweep is the instrument
# that can see both (the gate's CI checkout is katgpt-rs-only).
GATE_PINS = Path(__file__).resolve().parent / "numbering_floors.txt"

# DATA, not derived from the tree — see the module docstring's scope section.
SERIAL_DIRS = (".plans", ".issues", ".research", ".proposals")
ALL_DIRS = SERIAL_DIRS + (".benchmarks", ".docs")

# ONE list, read by the worktree advisory AND by the HEAD re-classification,
# and DERIVED from the directories above rather than typed beside them. It must
# name every file that can CHANGE a verdict: here that is any numbered document
# (a holder appearing or leaving moves `dup`, `above` and the `n_files` floor)
# and each `.highwater`, which `.../*` already covers — the allocator is a file
# in the directory it governs, so a second pattern for it would be a fifth
# hand-typed copy of the shape T5g measured going wrong three ways at once.
SCOPE = tuple(f"{d}/*" for d in ALL_DIRS)


def contract_repos(workspace: Path) -> list[Path]:
    """Derived population: a root BOUNDARY.md AND a .git DIRECTORY.

    The `.git` test is a directory test on purpose — a worktree's `.git` is a
    FILE, and a throwaway worktree of a repo already in the walk would be
    counted twice (the trap scripts/repo_set.txt's own derivation documents).
    Names pass through the machine-local alias codec (`repo_alias.py`) so the
    returned paths carry the CONTRACT spelling every tracked pin is keyed on.
    """
    return [workspace / n for n in repo_alias.apply(
        p.name for p in workspace.iterdir()
        if (p / "BOUNDARY.md").is_file() and (p / ".git").is_dir()
    )]


PIN_FIELDS = ("min_files", "max_dup", "max_above", "max_malformed",
              "max_resets", "min_numbers", "max_hist",
              "min_bench", "max_rename", "max_hand")


def parse_rows(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    want = 1 + len(PIN_FIELDS)
    # pins carry UTF-8 punctuation; the locale codec (cp1252 on Windows) cannot decode it (2026-09-06 4090-box catch)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != want:
            raise ValueError(f"malformed pin row (want {want} fields): {raw!r}")
        rows[parts[0]] = {k: int(v) for k, v in zip(PIN_FIELDS, parts[1:])}
    return rows


# The classified-class CAPS the sweep's own-repo row shares with the per-push
# gate's scalars. The classifier is shared by import (`ng.classify_stems`), so
# once both instruments classify the same repo, the ONLY way their verdicts for
# these classes can diverge is this pair of hand-maintained pin files drifting
# apart — measured 2026-10-29 both sides agree (0/59), and this keeps them
# agreeing. Floors are deliberately NOT synced: `min_bench` and the gate's
# `min_files.benchmarks` are headroom estimates under the same measured
# population and may legitimately differ (392 vs 400 at the 2026-10-29
# landing); equality on a floor is not a meaningful record, only equality on
# an exact measured cap is.
SYNC_CAP_PAIRS = (("max_rename", "max_rename_residue"),
                  ("max_hand", "max_hand_read"))


def classified_pin_sync(row: dict[str, int] | None, gate_pins: dict,
                        own: str) -> list[str]:
    """The sweep row for THIS repo vs the per-push gate's pins.

    Two instruments classify katgpt-rs's numbered directories with the SAME
    classifier (`numbering_gate.classify_stems`, imported above): the per-push
    gate (numbering_gate.py + numbering_floors.txt, CI) and this sweep
    (numbering_drift_floors.txt, workstation). Equal-by-construction held only
    while nobody looked; the 2026-10-29 gate port made the per-push half real,
    which made pin divergence a live verdict-splitting hazard — the sweep would
    red (or green) a state the gate calls the opposite, for no repo reason.
    This is the sync assertion: extracted so selftest can arm it (Issue 775's
    law — main() below invokes it unconditionally, on every run).

    Scope: the two CAPS only, and the gate's family lane EXISTING at all. A
    vanished `family_dirs` means the classified classes lost their per-push
    half silently — a divergence of scope, not of value, and just as verdict-
    splitting. Floors (`min_bench` vs `min_files.benchmarks`) are excluded by
    design — different slack conventions under the same measured population,
    not one record in two media (SYNC_CAP_PAIRS above).
    """
    flags: list[str] = []
    if row is None:
        flags.append(f"own-repo pin row missing for {own} in {PINS.name} — "
                     f"the sweep cannot verify its own classified caps "
                     f"against the gate's")
        return flags
    if "family_dirs" not in gate_pins:
        flags.append("gate pins declare NO family_dirs — the classified "
                     "classes lost their per-push half (numbering_floors.txt "
                     "scope regressed to serial-only); the CI-side wall is "
                     "gone and this sweep is the only instrument left")
        return flags
    for sweep_key, gate_key in SYNC_CAP_PAIRS:
        gv = gate_pins.get(gate_key)
        if not isinstance(gv, int):
            flags.append(f"gate pin {gate_key} is not an integer: {gv!r} — "
                         f"the gate itself would refuse this posture")
            continue
        if row[sweep_key] != gv:
            flags.append(
                f"{sweep_key}: sweep row pins {row[sweep_key]} ({PINS.name}) "
                f"!= gate pin {gate_key}={gv} (numbering_floors.txt) — ONE "
                f"measured record in two media; re-pin both in one commit")
    return flags


def audit(repo: Path) -> dict:
    """One repo -> its six finding classes + the two populations behind them.

    ⛔ `dup` and `hist` are NOT two views of one thing and must never be pooled
    (Issue 820). `dup` reads the WORKTREE through `tracked_paths`, so it sees a
    collision only while both holders are still on disk — and a document closed
    under the noise-reduction rule is DELETED, so the case where both have
    closed leaves nothing to read and the row is green. That is the MAJORITY
    case: measured 2026-09-17 over the derived population, `dup` totalled 0
    workspace-wide and `hist` totalled 183.

    `hist` is `numbering_gate.historical_collisions` — Issue 795's recovery,
    IMPORTED rather than re-derived, because the `-M` rename exclusion it
    carries is subtle enough that a second copy is a second thing to get wrong
    (Issue 755, and the reason the gate's own docstring gives for importing it
    from `citation_weight`). It ran in one repo of sixteen for two days: the
    gate is katgpt-rs-scoped by construction and this sweep — the cross-repo
    verdict half of that exact gate — never picked it up.

    ⚠ `hist` deliberately does NOT apply `dup`'s untracked split. An untracked
    holder is a colleague's WIP and not a defect *for a worktree class*; for a
    history class the other holder was REMOVED and was never going to be in
    anybody's worktree, so requiring two tracked copies would discard every row
    the class exists for. An untracked WIP beside a removed number is reported,
    and it is reported correctly — that number is about to be held twice.
    """
    tracked = ng.tracked_paths(repo, list(ALL_DIRS))
    dup, above, malformed, resets, unbumped = [], [], [], [], []
    collision, rename_res, hand_read = [], [], []
    n_serial = 0
    n_family = 0
    n_bench = 0
    grouped: list[tuple[str, dict]] = []
    for dirname in ALL_DIRS:
        by_num, hw, n, hw_bad = ng.scan(repo, dirname, tracked)
        grouped.append((dirname, by_num))
        if hw_bad is not None:
            malformed.append(f"{dirname}/.highwater = {hw_bad!r}")
        if dirname in (".benchmarks", ".docs"):
            n_bench += sum(1 for files in by_num.values()
                           for _nm, tr in files if tr)
        if dirname not in SERIAL_DIRS:
            continue                       # raw dup stays serial-scoped
        n_serial += n
        for num, files in sorted(by_num.items()):
            if sum(1 for _, tr in files if tr) < 2:
                continue                   # untracked = a colleague's WIP
            names = " · ".join(nm for nm, _ in files)
            dup.append(f"{dirname}/{num:03d} ×{len(files)}: {names}")
        if hw is not None and by_num and max(by_num) > hw:
            above.append(f"{dirname}: max {max(by_num)} > .highwater {hw}")
    # ── the classified classes (every dir, tracked holders only). The raw
    # `dup` above keeps its serial-only ratchet semantics and its existing
    # pins; these are the decidability upgrade — see the docstring's scope
    # section. A collision group of a FAMILY plus one token-carrying intruder
    # still fires: the pairwise token rule runs over every holder pair.
    for dirname, by_num in grouped:
        in_family_dir = dirname not in SERIAL_DIRS
        for num, files in sorted(by_num.items()):
            held = [(nm, tr) for nm, tr in files if tr]
            if len(held) < 2:
                continue
            stems = [ng.stem_of(nm) for nm, _tr in held]
            cls = ng.classify_stems(stems)
            names = " · ".join(nm for nm, _tr in held)
            row = f"{dirname}/{num} ×{len(held)}: {names}"
            if cls == "COLLISION":
                collision.append(row)
            elif cls == "RENAME-RESIDUE":
                rename_res.append(row)
            elif cls == "HAND-READ":
                hand_read.append(row)
            else:
                n_family += 1
                if in_family_dir:
                    continue               # the intended convention — silent
                # a serial-dir FAMILY is disclosed under dup's own row above;
                # nothing more to say here.
                continue
    # ── the counter-transition classes (Issues 769 + 770): every dir WITH a
    # counter, judged per-commit against its own parents (merge commits
    # included). The walker is the audit's, imported not re-implemented.
    for dirname in ALL_DIRS:
        hw_f = repo / dirname / ".highwater"
        if not hw_f.is_file():
            continue
        events = hca.counter_history(repo, dirname)
        if not events:
            continue                       # never committed — nothing to walk
        w = hca.classify_history(events, repo=repo)
        for a, b, h in w["resets"]:
            resets.append(f"{dirname}: {a}->{b} @ {h[:10]} "
                          f"({hca.commit_subject(repo, h)[:48]})")
        try:
            wt = int(hw_f.read_text(encoding="utf-8").strip().split()[-1])
        except (ValueError, IndexError, OSError):
            continue                       # malformed — ng's class owns it
        if w["final"] is not None and wt < w["final"]:
            unbumped.append(f"{dirname}: worktree {wt} < history max {w['final']} "
                            f"(checkout state — REPORT only)")
    # ── the HISTORY class (Issue 820). Its population is `n_numbers` and not
    # `n_files`, and the two break SEPARATELY, which is why both are floored:
    # `n_files` counts tracked documents on disk, `n_numbers` counts distinct
    # numbers recovered from `git log -M --diff-filter=D` PLUS disk. A git-log
    # regression leaves `n_files` untouched and collapses `n_numbers` to the
    # on-disk count, and every ceiling then passes green over a blind walk.
    hist_rows, n_numbers = ng.historical_collisions(repo, list(SERIAL_DIRS))
    hist = [f"{d}/{num:03d} x{len(stems)}: {' · '.join(stems)}"
            for d, num, stems in hist_rows]
    return {"dup": dup, "above": above, "malformed": malformed,
            "resets": resets, "unbumped": unbumped, "hist": hist,
            "collision": collision, "rename_residue": rename_res,
            "hand_read": hand_read, "n_family": n_family,
            "n_files": n_serial, "n_numbers": n_numbers, "n_bench": n_bench}


def head_listing(repo: Path) -> dict[str, list[str]] | None:
    """`{dirname: [direct child names]}` as HEAD carries them, or None.

    Issue 822 T5h. `None` is "this run has nothing committed to compare
    against" — not a repository, or `git ls-tree` refused — and it must stay
    distinguishable from an empty listing, which is a repository whose numbered
    directories are genuinely empty at HEAD. Pooling them would report every
    number in the repo as UNCOMMITTED.

    ⛔ **DIRECT children only, because `scan` iterates `d.iterdir()`.** `-r`
    recurses, so a nested path contributes its first component (the
    subdirectory entry `iterdir` would have yielded) and NOT its leaf: a
    `.benchmarks/806_x/fig.png` must not read as a numbered document named
    `fig.png`, and the directory `806_x` must not vanish from the count.

    `-z` rather than the default: git QUOTES a non-ASCII path in its plain
    output, and a quoted name is a different string from the one `iterdir`
    yields, which would split one document into two rows across the two sides.
    """
    if not (repo / ".git").is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "-z", "--name-only",
             "HEAD", "--", *ALL_DIRS],
            capture_output=True, encoding="utf-8", errors="replace", check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    names: dict[str, set[str]] = {d: set() for d in ALL_DIRS}
    for rel in out.split("\0"):
        rel = rel.strip()
        if not rel:
            continue
        dirname, _, rest = rel.partition("/")
        if dirname in names and rest:
            names[dirname].add(rest.split("/", 1)[0])
    return {d: sorted(v) for d, v in names.items()}


def head_audit(repo: Path) -> dict | None:
    """The three WORKTREE-derived classes + `n_files`, as HEAD carries them.

    Issue 822 T5h — and the class split is the whole design, so it is written
    down rather than implied:

    * `dup` / `above` / `malformed` / `n_files` read the worktree (a listing,
      and `.highwater`'s bytes). They are what a commit of this checkout would
      reproduce only by accident, so the PINS read this side.
    * `hist` / `resets` / `n_numbers` read `git log`. A dirty worktree cannot
      move them, so re-deriving them here would be a second copy of an
      identical answer.
    * ⛔ `unbumped` is a WORKTREE quantity BY CONSTRUCTION — "worktree N <
      history max M", the checkout-state class Issue 770 made report-only. It
      is the one row in this sweep that would be destroyed by adjudicating it
      to HEAD, where the comparison is vacuous. It is also unpinned, so it
      reaches no ceiling.

    ⚠ No `head_tree` and no skip branch, unlike T5g's sweep. This classifier
    reads only NAMES plus one small file per directory, so HEAD costs ~7 cheap
    git calls (measured: `ls-tree` over riir-ai's 1601 numbered paths, 0.05s)
    against the ~30s a materialised tree costs there. Running it
    unconditionally is what closes the UNTRACKED hole `markdown_fence` had to
    split by hand: an untracked file is in no `git status` diff and in no HEAD
    listing, so its row lands in UNCOMMITTED here by construction rather than
    by a second predicate somebody has to remember to write.
    """
    listing = head_listing(repo)
    if listing is None:
        return None
    tracked = {f"{d}/{n}" for d, ns in listing.items() for n in ns}
    dup, above, malformed = [], [], []
    collision, rename_res, hand_read = [], [], []
    n_serial = 0
    n_family = 0
    n_bench = 0
    grouped: list[tuple[str, dict]] = []
    for dirname in ALL_DIRS:
        by_num, n = ng.group_numbered(listing[dirname], dirname, tracked)
        grouped.append((dirname, by_num))
        hw, hw_bad = ng.parse_highwater(head_text(repo, f"{dirname}/{ng.HIGHWATER}"))
        if hw_bad is not None:
            malformed.append(f"{dirname}/.highwater = {hw_bad!r}")
        if dirname in (".benchmarks", ".docs"):
            n_bench += sum(1 for files in by_num.values()
                           for _nm, tr in files if tr)
        if dirname not in SERIAL_DIRS:
            continue
        n_serial += n
        for num, files in sorted(by_num.items()):
            if sum(1 for _, tr in files if tr) < 2:
                continue
            names = " · ".join(nm for nm, _ in files)
            dup.append(f"{dirname}/{num:03d} ×{len(files)}: {names}")
        if hw is not None and by_num and max(by_num) > hw:
            above.append(f"{dirname}: max {max(by_num)} > .highwater {hw}")
    for dirname, by_num in grouped:
        for num, files in sorted(by_num.items()):
            held = [(nm, tr) for nm, tr in files if tr]
            if len(held) < 2:
                continue
            stems = [ng.stem_of(nm) for nm, _tr in held]
            cls = ng.classify_stems(stems)
            names = " · ".join(nm for nm, _tr in held)
            row = f"{dirname}/{num} ×{len(held)}: {names}"
            if cls == "COLLISION":
                collision.append(row)
            elif cls == "RENAME-RESIDUE":
                rename_res.append(row)
            elif cls == "HAND-READ":
                hand_read.append(row)
            else:
                n_family += 1
    return {"dup": dup, "above": above, "malformed": malformed,
            "collision": collision, "rename_residue": rename_res,
            "hand_read": hand_read, "n_family": n_family,
            "n_files": n_serial, "n_bench": n_bench}


ADJUDICATED = ("dup", "above", "malformed",
               "collision", "rename_residue", "hand_read")


def adjudicate(repo: Path, got: dict) -> tuple[dict, dict]:
    """-> (`{class: HeadDelta}`, the dict `pin_flags` should read).

    The named seam, extracted rather than left inline in `main()` beside its
    own error messages — Issue 822 T5e's finding, which is that nothing
    automatic walls this join and only an arm aimed AT IT can.

    ⛔ **The row IS the key.** These rows are strings that already carry their
    own detail (`".issues/121 ×2: a · b"`), so a number whose HOLDER SET moved
    splits in both directions: the worktree's shape is UNCOMMITTED and HEAD's
    is MASKED. That is the same rule T5f and T5g reached from the other side —
    a key that omits a field the ceiling reads lets the worktree silently
    override HEAD — and here it costs nothing to get right, because there is no
    field to omit.

    A repo with no HEAD to read (`head_audit` → None) yields every row
    COMMITTED: the conservative direction for a bucket whose purpose is to
    WITHHOLD rows from a ceiling, and `split_rows`' own rule for an
    undeterminable address.
    """
    head = head_audit(repo)
    if head is None:
        return ({cls: HeadDelta(list(got[cls]), [], []) for cls in ADJUDICATED},
                got)
    deltas = {cls: delta_of(got[cls], head[cls], lambda r: r)
              for cls in ADJUDICATED}
    judged = dict(got)
    for cls in ADJUDICATED:
        judged[cls] = deltas[cls].head
    # ⚠ A FLOOR is a pin too, and Issue 797 measured this exact class on one:
    # another session's uncommitted deletion took a citation population from
    # 607 to 601, and a floor re-pinned from such a run bakes an in-flight edit
    # into a tracked file. `n_files` is the population behind `min_files`, so
    # it is HEAD's count that the floor adjudicates — and `n_bench` is the
    # population behind `min_bench` under exactly the same rule.
    judged["n_files"] = head["n_files"]
    judged["n_bench"] = head["n_bench"]
    return deltas, judged


def pin_flags(got: dict, row: dict[str, int]) -> list[str]:
    """One repo's audit + its pin row -> the failures, in reading order.

    EXTRACTED out of main() (Issue 820 T4). It sat inline between two calls
    into the classifier, so `selftest` could not reach it and the classifier's
    own arms structurally cannot: Issue 775's sentence, in the instrument that
    quotes it. An arm that re-derived the predicate instead would assert an
    abstraction and leave this path free to disagree with it.
    """
    flags = []
    if got["n_files"] < row["min_files"]:
        flags.append(f"population FLOOR breached: {got['n_files']} < {row['min_files']}")
    if got["n_bench"] < row["min_bench"]:
        flags.append(
            f"bench population FLOOR breached: {got['n_bench']} < {row['min_bench']} "
            f"— the classified classes (collision/rename/hand) are green over a "
            f"blind walk if this fires; a wall is only as good as the walk "
            f"beneath it")
    if got["collision"]:
        flags.append(
            f"COLLISION WALL: {len(got['collision'])} filename-decidable "
            f"two-owner number group(s) — never pinned, never a ratchet; "
            f"renumber the loser by the issue-556 protocol (first allocation "
            f"keeps, every citing site moves in the same commit): "
            + " | ".join(got["collision"]))
    if len(got["rename_residue"]) > row["max_rename"]:
        flags.append(
            f"rename residues {len(got['rename_residue'])} > pinned "
            f"{row['max_rename']} — one document under two spellings; the "
            f"content read decides which copy dies (git history keeps it)")
    if len(got["hand_read"]) > row["max_hand"]:
        flags.append(
            f"hand-read groups {len(got['hand_read'])} > pinned {row['max_hand']} "
            f"— a NEW token-less duplicate group; the ratchet catches the next "
            f"one, the standing backlog lives in the pins")
    if got["n_numbers"] < row["min_numbers"]:
        flags.append(
            f"HISTORY WALK floor breached: {got['n_numbers']} < {row['min_numbers']} "
            f"— `git log -M --diff-filter=D` recovered fewer numbers than the "
            f"removed-holder walk has ever seen; every ceiling below is green "
            f"over a blind instrument (Issue 820)")
    if len(got["hist"]) > row["max_hist"]:
        flags.append(
            f"historical collisions {len(got['hist'])} > pinned {row['max_hist']} "
            f"— a number is held by 2+ documents across HISTORY, which the "
            f"worktree `dup` column cannot see once both holders have closed "
            f"(Issues 795 + 820)")
    if len(got["dup"]) > row["max_dup"]:
        flags.append(f"duplicates {len(got['dup'])} > pinned {row['max_dup']}")
    if len(got["above"]) > row["max_above"]:
        flags.append(f"stale allocators {len(got['above'])} > pinned {row['max_above']}")
    if len(got["malformed"]) > row["max_malformed"]:
        flags.append(f"malformed allocators {len(got['malformed'])} > pinned {row['max_malformed']}")
    if len(got["resets"]) > row["max_resets"]:
        flags.append(
            f"counter resets {len(got['resets'])} > pinned {row['max_resets']} "
            f"— a commit (or merge resolution) landed below its parents' max: "
            f"the re-climb re-spends numbers (Issues 769+770)")
    return flags


def selftest() -> list[str]:
    """Pin the scope split and the row parser. Both fail SILENTLY otherwise."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        # a repo with one duplicate in a SERIAL dir and one in a FAMILY dir
        repo = ws / "fake-repo"
        (repo / ".plans").mkdir(parents=True)
        (repo / ".benchmarks").mkdir()
        for nm in ("001_a.md", "001_b.md"):
            (repo / ".plans" / nm).write_text("x", encoding="utf-8")
            (repo / ".benchmarks" / nm).write_text("x", encoding="utf-8")
        (repo / ".benchmarks" / ng.HIGHWATER).write_text("-n 7\n", encoding="utf-8")
        (repo / "BOUNDARY.md").write_text("x", encoding="utf-8")

        # tracked_paths() shells out to git; an untracked file is not a defect,
        # so a non-repo would report ZERO duplicates and pass vacuously. Pin
        # that the split is what decides, by driving audit() with a stub.
        real = ng.tracked_paths
        ng.tracked_paths = lambda r, d: {
            ".plans/001_a.md", ".plans/001_b.md",
            ".benchmarks/001_a.md", ".benchmarks/001_b.md",
        }
        try:
            got = audit(repo)
        finally:
            ng.tracked_paths = real

        if len(got["dup"]) != 1:
            fails.append(f"scope: expected 1 serial-dir duplicate, got {got['dup']}")
        if any(".benchmarks" in r for r in got["dup"]):
            fails.append("scope: a .benchmarks family number was reported as a duplicate")
        if len(got["malformed"]) != 1 or ".benchmarks" not in got["malformed"][0]:
            fails.append(f"scope: malformed check must cover family dirs too: {got['malformed']}")
        if got["n_files"] != 2:
            fails.append(f"population: counted {got['n_files']} serial files, expected 2")

        # the untracked split must still hold — one tracked copy is not a defect
        ng.tracked_paths = lambda r, d: {".plans/001_a.md"}
        try:
            got2 = audit(repo)
        finally:
            ng.tracked_paths = real
        if got2["dup"]:
            fails.append(f"untracked split broken: {got2['dup']}")

        # counter-transition classes (Issues 769 + 770): stub the walker's
        # history reader — a per-commit reset is reported with its commit; a
        # worktree below the history max is REPORTED (unbumped). The stub
        # spans BOTH arms: the Issue-770 review caught the landed version
        # restoring the real walker between them, so the in-flight arm read a
        # non-git tempdir and asserted nothing (mutation-proof: inverting the
        # guard passed clean).
        (repo / ".issues").mkdir(exist_ok=True)
        (repo / ".issues" / ng.HIGHWATER).write_text("3\n", encoding="utf-8")
        real_hist = hca.counter_history
        real_cls = hca.classify_history
        hca.counter_history = (
            lambda r, d: [{"hash": "h2", "value": 3, "base": 4, "kind": "reset"}]
            if d == ".issues" else [])
        hca.classify_history = (
            lambda ev, repo=None, is_ancestor=None:
            {"gaps": [], "duals": 0, "final": 4,
             "resets": [(4, 3, "h2")]})
        try:
            got3 = audit(repo)
            (repo / ".issues" / ng.HIGHWATER).write_text("5\n", encoding="utf-8")
            got4 = audit(repo)
        finally:
            hca.counter_history = real_hist
            hca.classify_history = real_cls
        if len(got3["resets"]) != 1 or "4->3" not in got3["resets"][0]:
            fails.append(f"reset: a per-commit reset must be reported: {got3['resets']}")
        if len(got3["unbumped"]) != 1 or "3 < history max 4" not in got3["unbumped"][0]:
            fails.append(f"unbumped: worktree 3 < history max 4 must report: {got3['unbumped']}")
        # a counter climbing in the worktree (in-flight) is NOT unbumped
        if got4["unbumped"]:
            fails.append(f"in-flight: worktree 5 > history max 4 must not report: {got4['unbumped']}")

        # population derivation: BOUNDARY.md + a .git DIR, both required
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        if [p.name for p in contract_repos(ws)] != []:
            fails.append("population: admitted a repo with no .git dir")
        (repo / ".git").mkdir()
        if [p.name for p in contract_repos(ws)] != ["fake-repo"]:
            fails.append("population: derivation is not BOUNDARY.md + .git dir")

        # ── the HISTORY class (Issue 820). `historical_collisions` shells out
        # to git and returns {} in a non-repo, so the fixture's two on-disk
        # `.plans/001_*` holders are the whole row set here — which is the arm
        # for the DISK half. Stub the git half to pin the union: a number whose
        # holders were both REMOVED is exactly the case `dup` cannot see, and
        # it must land in `hist` and NOT in `dup`.
        from citation_weight import removed_by_number as _real_rbn
        import citation_weight as _cw

        got5 = audit(repo)
        if len(got5["hist"]) != 1 or ".plans/001" not in got5["hist"][0]:
            fails.append(f"hist: two on-disk holders of one number must report: {got5['hist']}")
        if got5["n_numbers"] < 1:
            fails.append(f"hist: the history population must count the numbers walked: {got5['n_numbers']}")

        _cw.removed_by_number = (
            lambda r, d: {9: {"009_gone_a", "009_gone_b"}} if d == ".plans" else {})
        try:
            got6 = audit(repo)
        finally:
            _cw.removed_by_number = _real_rbn
        if not any(".plans/009" in h for h in got6["hist"]):
            fails.append(f"hist: a number whose holders were both REMOVED must "
                         f"report — this is the class `dup` cannot see: {got6['hist']}")
        if any("009" in d for d in got6["dup"]):
            fails.append(f"hist: a removed-only collision leaked into the tracked "
                         f"`dup` class — the two must never be pooled: {got6['dup']}")
        if got6["n_numbers"] <= got5["n_numbers"]:
            fails.append(f"hist: the history walk must widen the population it "
                         f"floors ({got6['n_numbers']} <= {got5['n_numbers']})")

        # row parser: 11 fields, comments stripped, arity enforced
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a\t10\t0\t0\t0\t0\t7\t2\t50\t0\t3  # trailing\n\n", encoding="utf-8")
        if parse_rows(pins) != {"repo-a": {"min_files": 10, "max_dup": 0,
                                           "max_above": 0, "max_malformed": 0,
                                           "max_resets": 0, "min_numbers": 7,
                                           "max_hist": 2, "min_bench": 50,
                                           "max_rename": 0, "max_hand": 3}}:
            fails.append("row parse: 11-field row not read correctly")
        # ⚠ a 10-field row is the PRE-classifier shape and must be REFUSED, not
        # read with the new columns defaulted: a defaulted `min_bench = 0` is a
        # floor that cannot fail, and `max_hand = 0` would red every repo with
        # a token-less bench family. Both directions of arity, so the parser
        # cannot only ever loosen.
        for bad_row in ("repo-a 1 2 3 4 5 6 7 8 9\n",
                        "repo-a 1 2 3 4 5 6 7 8 9 10 11\n"):
            pins.write_text(bad_row, encoding="utf-8")
            try:
                parse_rows(pins)
                fails.append(f"row parse: wrong-arity row accepted: {bad_row!r}")
            except ValueError:
                pass

        # ── the VERDICT arithmetic for the two new columns, which no
        # classifier reaches (Issue 775's rule). It calls `pin_flags`, the
        # function main() calls: re-deriving the predicate here would assert
        # an ABSTRACTION and leave the production path free to disagree with
        # it — the defect AGENTS.md records against injected probes, and the
        # reason `pin_flags` was extracted out of main() at all.
        row = {"min_files": 0, "max_dup": 9, "max_above": 9, "max_malformed": 9,
               "max_resets": 9, "min_numbers": 5, "max_hist": 1,
               "min_bench": 0, "max_rename": 9, "max_hand": 9}
        base = {k: [] for k in ("dup", "above", "malformed", "resets", "hist")}
        for n_hist, n_num, want in ((1, 5, False), (2, 5, True),
                                    (1, 4, True), (0, 5, False), (0, 4, True)):
            got_f = pin_flags({**base, "hist": ["x"] * n_hist,
                               "n_files": 0, "n_bench": 0, "collision": [],
                               "rename_residue": [], "hand_read": [],
                               "n_numbers": n_num}, row)
            if bool(got_f) != want:
                fails.append(f"verdict: hist={n_hist} nums={n_num} "
                             f"flags={got_f}, want fired={want}")
        # and the classes must be DISTINGUISHABLE in the message, or a reader
        # cannot tell a blind walk from a real new collision
        blind = pin_flags({**base, "n_files": 0, "n_bench": 0, "collision": [],
                           "rename_residue": [], "hand_read": [],
                           "n_numbers": 0}, row)
        grew = pin_flags({**base, "hist": ["x", "y"], "n_files": 0,
                          "n_bench": 0, "collision": [], "rename_residue": [],
                          "hand_read": [], "n_numbers": 9}, row)
        if not (blind and "WALK floor" in blind[0]):
            fails.append(f"verdict: a blind history walk must name itself: {blind}")
        if not (grew and "historical collisions" in grew[0]):
            fails.append(f"verdict: a new collision must name itself: {grew}")

        # ── the classified classes: the classifier's vectors first (pure, no
        # git), then the WALL's verdict arithmetic through pin_flags — the
        # function main() calls, never a re-derivation (the arm above's rule).
        # The vectors encode riir-train issue-556's arms INCLUDING the four
        # live collisions this sweep's landing resolved, so the two classifiers
        # are pinned to agree by data, not by comment.
        vec = [
            (("issue771_flash_cmma", "issue772_expand_l2_staged_sum"), "COLLISION"),  # riir-ai 809
            (("issue879_t3_kv_weight_quant_nll", "issue880_t0_warm_read_stall"), "COLLISION"),  # riir-ai 872
            (("issue1127_t4_recell_bundle", "issue1132_metal_launch_floor_census_w1"), "COLLISION"),  # riir-ai 1008
            (("plan619_clef_text_ane_latency", "plan621_gdn_l8_eager_prefill"), "COLLISION"),  # riir-infer 884
            (("421_a", "421_b", "421_c"), "HAND-READ"),                  # the token-less family shape
            (("plan421_p05", "plan421_p2", "plan421_p3"), "FAMILY-EXPLICIT"),
            (("issue663_a", "issue663_b"), "FAMILY-EXPLICIT"),
            (("549_plan354_p2_t1", "549_plan354_p2t1"), "RENAME-RESIDUE"),  # the 556 founding residue
            (("plan999_x", "issue777_y"), "COLLISION"),                  # cross-kind disjoint tokens
            (("x", "y"), "HAND-READ"),
            (("plan421_a", "plan421_b", "plan888_z"), "COLLISION"),      # a token-carrying intruder inside a family
            (("421_a", "421_b", "999_z"), "HAND-READ"),                  # a TOKEN-LESS intruder is undecidable, like its family
            (("plan300_t1", "plan300_t2", "issue300_extra"), "COLLISION"),  # plan300 vs issue300 differ
        ]
        for stems_v, want_cls in vec:
            got_cls = ng.classify_stems(list(stems_v))
            if got_cls != want_cls:
                fails.append(f"classify: {stems_v} -> {got_cls}, want {want_cls}")
        # the stem splitter itself: the whole-filename-match trap (see the
        # stem_of docstring) yields "" for every file, classifying every
        # group RENAME-RESIDUE and greenning every wall — this arm caught it
        if ng.stem_of("809_issue771_flash_cmma.md") != "issue771_flash_cmma":
            fails.append("stem_of: whole-filename match trap: "
                         f"{ng.stem_of('809_issue771_flash_cmma.md')!r}")
        if ng.stem_of("0841_x.md") != "x" or ng.stem_of("0421_a.md") != "a":
            fails.append("stem_of: multi-digit prefix or plain-stem handling broke")
        wall_row = {"min_files": 0, "max_dup": 0, "max_above": 0,
                    "max_malformed": 0, "max_resets": 0, "min_numbers": 0,
                    "max_hist": 0, "min_bench": 0, "max_rename": 0,
                    "max_hand": 0}
        wall_base = {**{k: [] for k in base}, "n_files": 0, "n_bench": 50,
                     "n_numbers": 9, "collision": ["x"], "rename_residue": [],
                     "hand_read": []}
        wf = pin_flags(wall_base, wall_row)
        if not any("COLLISION WALL" in f for f in wf):
            fails.append(f"verdict: the collision wall must fire through "
                         f"pin_flags regardless of every pin: {wf}")
        # and the bench population floor through the same seam
        ff = pin_flags({**wall_base, "collision": [], "n_bench": 1},
                       {**wall_row, "min_bench": 50})
        if not any("bench population FLOOR" in f for f in ff):
            fails.append(f"verdict: a bench-walk regression must red the "
                         f"min_bench floor: {ff}")

        # ── the classified-cap SYNC against the gate's pins (2026-10-29):
        #    one measured record in two media; divergence is itself the defect.
        sync_row = {"max_rename": 0, "max_hand": 59}
        sync_gp = {"family_dirs": ".benchmarks", "max_rename_residue": 0,
                   "max_hand_read": 59}
        if classified_pin_sync(sync_row, sync_gp, "katgpt-rs") != []:
            fails.append("pin-sync: agreeing caps must not flag")
        sf = classified_pin_sync({**sync_row, "max_hand": 57}, sync_gp, "r")
        if len(sf) != 1 or "57" not in sf[0] or "59" not in sf[0]:
            fails.append(f"pin-sync: hand divergence must name both sides: {sf}")
        sf = classified_pin_sync({**sync_row, "max_rename": 1}, sync_gp, "r")
        if len(sf) != 1 or "max_rename" not in sf[0]:
            fails.append(f"pin-sync: rename divergence must flag: {sf}")
        if len(classified_pin_sync(None, sync_gp, "r")) != 1:
            fails.append("pin-sync: missing own-repo row must flag")
        sf = classified_pin_sync(
            sync_row, {k: v for k, v in sync_gp.items() if k != "family_dirs"}, "r")
        if len(sf) != 1 or "family_dirs" not in sf[0]:
            fails.append(f"pin-sync: a vanished gate family lane must flag: {sf}")
        sf = classified_pin_sync(
            sync_row, {"family_dirs": ".benchmarks"}, "r")
        if (len(sf) != 2 or "max_rename_residue" not in sf[0]
                or "max_hand_read" not in sf[1]):
            fails.append(f"pin-sync: missing gate scalars flag PER PAIR: {sf}")
        sf = classified_pin_sync(
            sync_row, {"family_dirs": ".benchmarks", "max_rename_residue": 0}, "r")
        if len(sf) != 1 or "max_hand_read" not in sf[0] or "not an integer" not in sf[0]:
            fails.append(f"pin-sync: one missing gate scalar flags once: {sf}")
        sf = classified_pin_sync(
            sync_row, {**sync_gp, "max_hand_read": "junk"}, "r")
        if len(sf) != 1 or "not an integer" not in sf[0]:
            fails.append(f"pin-sync: non-integer gate pin must flag: {sf}")

    return fails + adjudicate_arms()


def adjudicate_cases() -> list[str]:
    """`head_audit` / `adjudicate` end to end against REAL git — Issue 822 T5h.

    In `selftest` and behind no flag: this sweep has no `--canary`, and an arm
    that only runs when somebody types a flag runs on no invocation anybody
    makes (Issue 789's finding, measured on eight arms that had landed the day
    before). They cost ~1s.

    Stubbing git here would assert an abstraction and leave the production path
    free to disagree with it — which is the defect this file's own
    `pin_flags` extraction exists to avoid, one seam over. Every arm drives a
    real repository through a real commit.
    """
    import tempfile

    fails: list[str] = []

    def git(root, *args):
        subprocess.run(("git", "-C", str(root)) + args,
                       capture_output=True, check=True)

    def fixture(td: str):
        repo = Path(td) / "r"
        (repo / ".issues").mkdir(parents=True)
        (repo / ".issues" / "001_a.md").write_text("x", encoding="utf-8")
        (repo / ".issues" / ng.HIGHWATER).write_text("1\n", encoding="utf-8")
        git(repo.parent, "init", "-q", "r")
        git(repo, "config", "user.email", "arm@example.invalid")
        git(repo, "config", "user.name", "arm")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
        return repo

    def run(repo):
        got = audit(repo)
        return (got,) + adjudicate(repo, got)

    # a. A STAGED second holder is tracked, so the worktree reports a duplicate
    #    — and no commit contains it. This is the whole issue: without the
    #    split, `max_dup = 0` reds on another session's in-flight `git add`.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / "001_b.md").write_text("y", encoding="utf-8")
        git(repo, "add", ".issues/001_b.md")
        got, deltas, judged = run(repo)
        if len(got["dup"]) != 1:
            fails.append(f"arm a: the fixture is INERT — a staged second "
                         f"holder must be a worktree duplicate ({got['dup']})")
        if len(deltas["dup"].uncommitted) != 1 or judged["dup"]:
            fails.append(f"adjudicate: a STAGED-only duplicate reached the "
                         f"ceiling ({judged['dup']}) instead of UNCOMMITTED "
                         f"({deltas['dup'].uncommitted})")

    # b. The silent direction: a COMMITTED duplicate that an uncommitted
    #    deletion hides. The worktree says clean and the pin must still red.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / "001_b.md").write_text("y", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "dup")
        (repo / ".issues" / "001_b.md").unlink()
        got, deltas, judged = run(repo)
        if got["dup"]:
            fails.append(f"arm b: the fixture is INERT — the worktree must "
                         f"read clean ({got['dup']})")
        if len(deltas["dup"].masked) != 1 or len(judged["dup"]) != 1:
            fails.append(f"adjudicate: a COMMITTED duplicate hidden by this "
                         f"worktree was not MASKED ({deltas['dup']}) — a "
                         f"committed collision reported clean is the silent "
                         f"direction, and `.head` is what the pin reads")

    # c. ⛔ The UNTRACKED half, which `markdown_fence` had to split by hand.
    #    An untracked numbered file is in `git status --porcelain
    #    --untracked-files=no` NOWHERE, so a dirty-set-scoped instrument cannot
    #    see it — and it still pushes `max` above `.highwater`. Running the
    #    HEAD pass unconditionally is what closes it: HEAD's listing simply has
    #    no such file.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / "009_wip.md").write_text("z", encoding="utf-8")
        got, deltas, judged = run(repo)
        if len(got["above"]) != 1:
            fails.append(f"arm c: the fixture is INERT — an untracked 009 "
                         f"above .highwater 1 must be a worktree stale-"
                         f"allocator row ({got['above']})")
        if judged["above"] or len(deltas["above"].uncommitted) != 1:
            fails.append(f"adjudicate: an UNTRACKED file's stale-allocator "
                         f"row reached the ceiling ({judged['above']}) — a "
                         f"finding on a file `git log` cannot see, counted "
                         f"against a pin, is this issue's whole subject")

    # d. `.highwater` in BOTH directions, because it is one small file whose
    #    bytes decide two classes and every session edits it.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / ng.HIGHWATER).write_text("-n 5\n", encoding="utf-8")
        got, deltas, judged = run(repo)
        if len(got["malformed"]) != 1 or judged["malformed"]:
            fails.append(f"adjudicate: an uncommitted malformed allocator "
                         f"reached the ceiling ({judged['malformed']})")
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / ng.HIGHWATER).write_text("-n 5\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "bad hw")
        (repo / ".issues" / ng.HIGHWATER).write_text("5\n", encoding="utf-8")
        got, deltas, judged = run(repo)
        if got["malformed"] or len(judged["malformed"]) != 1:
            fails.append(f"adjudicate: a COMMITTED malformed allocator that "
                         f"this worktree repairs must stay on the ceiling "
                         f"({judged['malformed']}) — the repair is not landed "
                         f"until it is committed (Issue 798)")

    # e. A FLOOR is a pin too (Issue 797's measured case was a population, not
    #    a finding): `min_files` adjudicates HEAD's count, never the worktree's.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".issues" / "002_new.md").write_text("q", encoding="utf-8")
        git(repo, "add", "-A")
        got, deltas, judged = run(repo)
        if got["n_files"] != 2 or judged["n_files"] != 1:
            fails.append(f"adjudicate: the population floor read the worktree "
                         f"({got['n_files']}) instead of HEAD "
                         f"({judged['n_files']}) — a floor re-pinned from such "
                         f"a run bakes another session's edit into a tracked "
                         f"file")

    # f. DIRECT children only, because `scan` iterates `iterdir()`: a nested
    #    artifact contributes the DIRECTORY entry, never its leaf. Without
    #    this the two sides disagree about what a document even is.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".benchmarks" / "806_x").mkdir(parents=True)
        (repo / ".benchmarks" / "806_x" / "007_fig.md").write_text(
            "f", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "nested")
        listing = head_listing(repo)
        if listing is None or listing[".benchmarks"] != ["806_x"]:
            fails.append(f"head_listing: a nested path must contribute its "
                         f"DIRECTORY entry and not its leaf: {listing}")

    # g. Not a repository -> every row COMMITTED, never a fabricated empty
    #    HEAD, which would report the whole repo as UNCOMMITTED and silence
    #    every ceiling in it.
    with tempfile.TemporaryDirectory() as td:
        plain = Path(td) / "plain"
        (plain / ".issues").mkdir(parents=True)
        (plain / ".issues" / "001_a.md").write_text("x", encoding="utf-8")
        if head_audit(plain) is not None:
            fails.append("head_audit: a non-repository produced a HEAD view")
        got = {"dup": ["r"], "above": [], "malformed": [], "n_files": 1,
               "collision": [], "rename_residue": [], "hand_read": [],
               "n_bench": 0}
        deltas, judged = adjudicate(plain, got)
        if judged["dup"] != ["r"] or deltas["dup"].uncommitted:
            fails.append(f"adjudicate: a repo with no HEAD must leave every "
                         f"row COMMITTED ({deltas['dup']})")

    # h. A STAGED token-disjoint collision in .benchmarks — the class the raw
    #    dup never saw. Tracked, so the worktree reports it; no commit
    #    contains it, so the wall must NOT red: a colleague's in-flight renumber
    #    is exactly the shape the UNCOMMITTED split exists for.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".benchmarks").mkdir()
        (repo / ".benchmarks" / ng.HIGHWATER).write_text("2\n", encoding="utf-8")
        (repo / ".benchmarks" / "010_plan1_a.md").write_text("x", encoding="utf-8")
        (repo / ".benchmarks" / "010_plan2_b.md").write_text("y", encoding="utf-8")
        git(repo, "add", ".benchmarks")
        got, deltas, judged = run(repo)
        if not got["collision"]:
            fails.append(f"arm h: the fixture is INERT — a staged token-"
                         f"disjoint benchmark pair must classify COLLISION "
                         f"({got['collision']})")
        if judged["collision"] or len(deltas["collision"].uncommitted) != 1:
            fails.append(f"adjudicate: a STAGED collision reached the wall "
                         f"({judged['collision']}) instead of UNCOMMITTED")

    # i. The COMMITTED collision reds the wall through pin_flags even with
    #    every ceiling at zero-and-generous — and a committed FAMILY beside it
    #    does not. This is the two-sided non-vacuity arm for the whole class.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".benchmarks").mkdir()
        (repo / ".benchmarks" / ng.HIGHWATER).write_text("3\n", encoding="utf-8")
        (repo / ".benchmarks" / "011_plan1_a.md").write_text("x", encoding="utf-8")
        (repo / ".benchmarks" / "011_plan2_b.md").write_text("y", encoding="utf-8")
        (repo / ".benchmarks" / "012_plan3_c.md").write_text("z", encoding="utf-8")
        (repo / ".benchmarks" / "012_plan3_d.md").write_text("w", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "coll+family")
        got, deltas, judged = run(repo)
        if len(judged["collision"]) != 1:
            fails.append(f"arm i: a COMMITTED collision must reach the wall "
                         f"({judged['collision']})")
        if got["n_family"] != 1:
            fails.append(f"arm i: the fixture is INERT — the committed family "
                         f"must count as FAMILY-EXPLICIT (n_family="
                         f"{got['n_family']})")
        flags = pin_flags(judged, {"min_files": 0, "max_dup": 0,
                                   "max_above": 0, "max_malformed": 0,
                                   "max_resets": 0, "min_numbers": 0,
                                   "max_hist": 0, "min_bench": 0,
                                   "max_rename": 0, "max_hand": 0})
        if not any("COLLISION WALL" in f for f in flags):
            fails.append(f"arm i: the wall must red through pin_flags on a "
                         f"committed collision with every ceiling 0: {flags}")
        if any("max_dup" in f for f in flags) or any("hand" in f for f in flags):
            fails.append(f"arm i: a family plus a collision must not leak into "
                         f"the raw dup or hand-read ceilings: {flags}")

    # j. min_bench is a floor, and a floor is a pin too (Issue 797): it reads
    #    HEAD's count, never the worktree's.
    with tempfile.TemporaryDirectory() as td:
        repo = fixture(td)
        (repo / ".benchmarks").mkdir()
        (repo / ".benchmarks" / "020_plan9_only.md").write_text("x", encoding="utf-8")
        git(repo, "add", "-A")
        got, deltas, judged = run(repo)
        if got["n_bench"] != 1 or judged["n_bench"] != 0:
            fails.append(f"adjudicate: the bench population floor read the "
                         f"worktree ({got['n_bench']}) instead of HEAD "
                         f"({judged['n_bench']})")

    return fails


def adjudicate_arms() -> list[str]:
    """The cases above, plus the STUB PROBE that proves they sit on the seam.

    ⛔ Aimed at `delta_of` — the helper `adjudicate` ACTUALLY calls — because
    two of this family's first three probes reported a false all-clear by
    being aimed at a function the target never invokes. A `delta_of` that
    files everything as COMMITTED is exactly the regression the wiring exists
    to prevent, and the cases above must red on it.

    ⚠ `n_files` is deliberately not counted here: it comes from `head_audit`
    and never passes through `delta_of`, so an arm that expected the probe to
    red it would be asserting the stub rather than the seam.
    """
    fails = adjudicate_cases()
    real = globals()["delta_of"]
    globals()["delta_of"] = lambda wt, hd, key: HeadDelta(list(wt), [], [])
    try:
        probed = adjudicate_cases()
    finally:
        globals()["delta_of"] = real
    if len(probed) < 4:
        fails.append(f"STUB PROBE: a `delta_of` that files every row as "
                     f"COMMITTED red only {len(probed)} of the provenance "
                     f"arms — they are not sitting under the seam")
    return fails


def main() -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    fails = selftest()
    if fails:
        print("✗ numbering sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    if "--measure-bench" in sys.argv:
        # Early-exit measurement mode: paste-ready values for the three new
        # pin columns, BEFORE the pins exist to be read (the landing run's
        # chicken-and-egg). Worktree counts; a landing re-pins from HEAD
        # verdicts, but the two agree on a clean tree by construction.
        repos_m = contract_repos(WORKSPACE)
        if not repos_m:
            print("✗ empty population")
            return 2
        for repo_m in repos_m:
            path_m = open_repo(repo_m.name, WORKSPACE)
            got_m = audit(path_m)
            print(f"{repo_m.name}\tbench={got_m['n_bench']}\t"
                  f"rename={len(got_m['rename_residue'])}\t"
                  f"hand={len(got_m['hand_read'])}\t"
                  f"fam={got_m['n_family']}\tcoll={len(got_m['collision'])}")
        return 0
    try:
        pins = parse_rows(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    # The sweep's own-repo classified caps and the per-push gate's pins are ONE
    # measured record in two media — refuse loud on divergence BEFORE any repo
    # verdict (a sweep computing this repo's verdicts against a stale cap would
    # disagree with the CI gate for no repo reason). Single home: the sweep is
    # the side that reads both files; the gate needs no mirror of this.
    if GATE_PINS.is_file():
        own = repo_alias.apply((REPO_ROOT.name,))[0]
        sync_flags = classified_pin_sync(
            pins.get(own), ng.parse_pins(GATE_PINS), own)
    else:
        sync_flags = [f"gate pins file missing: {GATE_PINS} — the "
                      f"classified-cap sync cannot run"]
    if sync_flags:
        print("✗ classified-cap SYNC between the two numbering instruments "
              "FAILED — this sweep's verdicts for its own repo would not "
              "match the per-push gate's:")
        for f in sync_flags:
            print(f"    {f}")
        return 2

    show_hist = "--hist-rows" in sys.argv

    repos = contract_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    seen = {p.name for p in repos}
    bad = False
    tot_dup = tot_above = tot_mal = tot_reset = tot_unb = tot_hist = 0
    tot_coll = tot_rename = tot_hand = tot_fam = 0
    tot_uncommitted = tot_masked = 0

    for repo in repos:
        # Issue 842: the derived handle is CONTRACT-named; the DIRECTORY is
        # the on-disk spelling. Audit the real checkout, label by the handle.
        path = open_repo(repo.name, WORKSPACE)
        got = audit(path)
        # Issue 822 — the DISPLAY reads the worktree (it is what the files say
        # today, and hiding that would be its own lie); every CEILING and the
        # `min_files` FLOOR read what a commit of this checkout would produce.
        deltas, judged = adjudicate(path, got)
        held = {r for d in deltas.values() for r in d.uncommitted}
        tot_uncommitted += sum(len(d.uncommitted) for d in deltas.values())
        tot_masked += sum(len(d.masked) for d in deltas.values())
        row = pins.get(repo.name)
        tot_hist += len(got["hist"])
        tot_dup += len(got["dup"])
        tot_above += len(got["above"])
        tot_mal += len(got["malformed"])
        tot_reset += len(got["resets"])
        tot_unb += len(got["unbumped"])
        tot_coll += len(got["collision"])
        tot_rename += len(got["rename_residue"])
        tot_hand += len(got["hand_read"])
        tot_fam += got["n_family"]
        # Issue 821: an acknowledged known-extra owes no pin row. This landed
        # here first (Issue 820 T6) as a local copy; it is `pin_row_exempt`
        # now, shared with the other 14 sweeps — the whole point of 821 being
        # that a rule in one instrument is a rule that has not generalised.
        if row is None:
            flags = ([] if pin_row_exempt(repo.name)
                     else ["UNPINNED — add a row (or it can never red)"])
        else:
            flags = pin_flags(judged, row)
        # unbumped is REPORT-ONLY (Issue 770 finding 4): a checkout-state
        # quantity is a function of which branch the box carries — printed
        # like the citation sweep's undecided rows, never pinned, never red.
        status = "✗" if flags else ("·" if (got["dup"] or got["above"] or got["malformed"]
                                    or got["resets"] or got["unbumped"]
                                    or got["collision"] or got["rename_residue"]
                                    or got["hand_read"]) else "✓")
        n_masked = sum(len(d.masked) for d in deltas.values())
        split = ""
        if held or n_masked:
            split = (f" [{len(held)} uncommitted"
                     + (f", {n_masked} MASKED" if n_masked else "") + "]")
        if judged["n_files"] != got["n_files"]:
            split += f" [files at HEAD={judged['n_files']}]"
        print(f"{status} {repo.name:22s} files={got['n_files']:<5d} nums={got['n_numbers']:<5d} "
              f"dup={len(got['dup'])} hist={len(got['hist'])} "
              f"stale={len(got['above'])} malformed={len(got['malformed'])} "
              f"resets={len(got['resets'])} unbumped={len(got['unbumped'])} "
              f"bench={got['n_bench']:<4d} coll={len(got['collision'])} "
              f"ren={len(got['rename_residue'])} hand={len(got['hand_read'])} "
              f"fam={got['n_family']}{split}")

        def mark(r: str) -> str:
            return " [UNCOMMITTED — not adjudicated]" if r in held else ""
        # ⚠ hist rows are printed only when the pin BREACHES, or on demand via
        # --hist-rows. The standing backlog is 183 rows workspace-wide and
        # dumping it every run is how a sweep stops being read (the cries-wolf
        # failure docs_gate.sh's preamble records); it lives in the pin file's
        # comments, which is where a backlog is adjudicated from.
        if show_hist or (row is not None and len(got["hist"]) > row["max_hist"]):
            for r in got["hist"]:
                print(f"      historical: {r}")
        for r in got["malformed"]:
            print(f"      malformed:  {r}{mark(r)}")
        for r in got["above"]:
            print(f"      stale:      {r}{mark(r)}")
        for r in got["dup"]:
            print(f"      duplicate:  {r}{mark(r)}")
        # Collision + residue rows print ALWAYS: the wall expects zero, and a
        # row nobody can see is the masked-direction failure mode. Hand-read
        # rows print on breach only — the workspace backlog is ~150 groups of
        # old-convention bench families, and printing them every run is the
        # cries-wolf failure (the --hist-rows reasoning, one class over).
        for r in got["collision"]:
            print(f"      ⛔ COLLISION: {r}{mark(r)}")
        for r in got["rename_residue"]:
            print(f"      ⛔ rename-residue: {r}{mark(r)}")
        if row is not None and len(got["hand_read"]) > row["max_hand"]:
            for r in got["hand_read"]:
                print(f"      hand-read:  {r}{mark(r)}")
        # A MASKED row is NOT in the worktree buckets — that is what MASKED
        # means — so it prints from the HEAD side or it prints nowhere, and a
        # ceiling reds over a row nobody can see.
        for cls in ADJUDICATED:
            for r in deltas[cls].masked:
                print(f"      ⛔ {cls:<10s} {r}  [MASKED — committed, hidden "
                      f"by this worktree]")
        for r in got["resets"]:
            print(f"      reset:      {r}")
        for r in got["unbumped"]:
            print(f"      unbumped:   {r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

        # ── T3: katgpt-rs does not RESTATE its own gate, it CROSS-CHECKS it.
        # This sweep imports the gate's `historical_collisions`, so comparing
        # the two counts would be true by construction — a pin that restates
        # its own input cannot fail (the full_gate Layer 2c rule). What is NOT
        # automatic is the gate's VERDICT over those rows: the ≥ era_boundary
        # MEMBERSHIP wall and the legacy RATCHET live in
        # number_collisions_expected.txt, which this sweep never reads
        # otherwise, so a pin gone stale there was invisible here.
        # `trap_sentinel_drift_sweep` vs `trap_sentinel_gate.POPULATION_FLOOR`,
        # one instrument over: where a sweep re-states a quantity its per-push
        # gate owns, it ASSERTS the two agree rather than trusting them.
        if repo.name == REPO_ROOT.name:
            try:
                rows, _ = ng.historical_collisions(repo, list(SERIAL_DIRS))
                scalars, cpins = ng.parse_collision_pins(
                    REPO_ROOT / "scripts" / "number_collisions_expected.txt")
                gate_fails, _notes = ng.collision_verdict(rows, scalars, cpins)
            except (OSError, ValueError, KeyError) as e:
                bad = True
                print(f"      ✗ gate cross-check UNREADABLE: {e} — the "
                      f"delegation is asserted, never assumed")
            else:
                for gf in gate_fails:
                    bad = True
                    print(f"      ✗ numbering_gate would RED on this: {gf}")

    # The population axis, shared (Issue 793): UNREGISTERED reds in every
    # posture, UNSEEN reds without the marker, and the same set DEFERS loudly
    # with it. Never auto-detected — a genuine removal whose row update was
    # forgotten is set-identical to a partial clone from the walk alone.
    pop_lines, deferred, pop_fail = population_verdict(pins, seen)

    # Issue 797 — the worktree is not the repo. This run reads files that
    # concurrent sessions are editing, so a finding may sit on a line no
    # commit contains. ADVISORY, never a failure: a sweep that hard-reds on
    # an ordinary dirty worktree is a sweep nobody runs. It rides the FINAL
    # line in BOTH directions (the `deferred` precedent) and is SILENT
    # unless the dirty set meets this sweep's own population — the numbered directories + the HISTORY headings the oracle reads.
    # ⛔ This call carried a hand-typed FIFTH copy of the population —
    # `(".issues/*", ".plans/*", ".docs/*", ".research/*", ".proposals/*",
    # "HISTORY.md")` — and it disagreed with the classifier in both directions:
    # `.benchmarks/*` was MISSING (it is in ALL_DIRS, and its `.highwater`
    # feeds `malformed` and `resets`), while `HISTORY.md` is read by NOTHING
    # here — the history classes come from `git log`, not from that document.
    # T5g measured the same shape one sweep over; `SCOPE` is why there is no
    # sixth copy.
    deferred.extend(sweep_advisory(
        seen, SCOPE, root=WORKSPACE,
        uncommitted_rows=tot_uncommitted, masked_rows=tot_masked))
    for _line in pop_lines:
        print(_line)
    if pop_fail:
        bad = True

    print(f"\n{len(repos)} contract repo(s) · {tot_dup} tracked duplicate(s) · "
          f"{tot_hist} historical collision(s) · "
          f"{tot_above} stale allocator(s) · {tot_mal} malformed allocator(s) · "
          f"{tot_reset} counter reset(s) · {tot_unb} un-bumped counter(s) [report-only]\n")
    print(f"  classified (all six dirs, tracked .md holders): "
          f"{tot_coll} COLLISION · {tot_rename} rename-residue · "
          f"{tot_hand} hand-read · {tot_fam} family-explicit")
    print(f"  ⛔ `tracked duplicate` and `historical collision` are NOT two "
          f"views of one quantity (Issue 820). The first reads the worktree, "
          f"so it goes green the moment both holders CLOSE — and every number "
          f"here is expected to end up removed under the noise-reduction rule. "
          f"The second recovers them from `git log -M --diff-filter=D` (Issue "
          f"795) and is the majority case — {tot_dup} vs {tot_hist} on THIS "
          f"run, derived, because a landing figure typed here would be a claim "
          f"about history rather than a measurement. Rows print when a pin "
          f"breaches, or on demand with --hist-rows.")
    print(f"  resets are judged per-commit against each commit's OWN parents "
          f"(Issue 770: the 769 walker's date-ordered interleaving manufactured "
          f"phantoms and never saw merges; this walk blames the commit that did "
          f"it — merge resolutions taking the lower side included). Pinned at "
          f"measured as a ratchet — a NEW reset reds at its pin. unbumped rows "
          f"are checkout state (branch-dependent), printed, never gated.")
    # Say the scope at the point of READING, not only in the docstring. The
    # raw `dup` ratchet is still SERIAL-scoped (its pins were measured there),
    # but the classified classes cover every numbered directory: the
    # "undecidable convention" excuse retired when the stem classifier landed
    # (2026-10-29) — owner identity IS decidable from the filenames.
    family = "/".join(d for d in ALL_DIRS if d not in SERIAL_DIRS)
    print(f"  scope: the RAW duplicate + stale checks cover {'/'.join(SERIAL_DIRS)} "
          f"only; the classified classes (COLLISION wall, rename-residue, "
          f"hand-read) cover ALL numbered dirs incl. {family} — a `coll=0` "
          f"above is a wall over every directory, a `dup=0` is NOT a claim "
          f"about {family}. See {PINS.name} for the measured pins.")
    if bad:
        print("✗ numbering sweep FAILED — see the ✗ rows above")
        for _d in deferred:
            print(f"  {deferral_line(_d)}")
        return 1
    _line = "✓ numbering sweep PASSED — nothing above its pinned ratchet"
    if deferred:
        _line += "; DEFERRED: " + "; ".join(deferred)
    print(_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
