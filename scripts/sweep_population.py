#!/usr/bin/env python3
"""The workstation sweeps' POPULATION verdict — one copy (Issue 793).

Every drift sweep in `scripts/` pins per-repo rows and then has to answer the
same question about the rows it did NOT see: is a pinned repo missing because
it was retired and somebody forgot the row, or because this box carries a
subset of the workspace? Those are set-identical from the walk alone, which is
why the answer is an **explicit marker** and never an inference
(`DOCS_GATE_PARTIAL_CLONE=1`, Issue 765).

Seven sweeps carried this loop, byte-identical, copy-pasted:

    for name in sorted(set(pins) - present):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

and hard-red on a known partial-clone box with every content assertion green.
A sweep that always reds is a sweep nobody runs, and its findings go unread
with it — measured: the percentile sweep's Issue-777 findings (a fabricated
floor, a correctly-shaped defect at the wrong address) sat behind four of these
reds for as long as the rows had existed. The raw remedy text is also the
dangerous one: "drop the row in that commit", offered on the box least
qualified to decide that, deletes live repos from the canonical set.

Three verdicts, and they are not interchangeable:

- **UNREGISTERED** — on this box, absent from `repo_set.txt`. A repo JOINING
  the workspace. Reds in EVERY posture; no amount of partial checkout explains
  a directory that is right there.
- **UNSEEN** — pinned (or in the snapshot) and not on this box, WITHOUT the
  marker. Never a pass: the instrument could not measure a repo it claims to
  cover.
- **DEFERRED** — the same set, WITH the marker. Reported loudly on the sweep's
  final line, so a green never reads as a claim about repos this run never
  touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skill_repo_set_gate import (  # noqa: E402
    KNOWN_EXTRA_MARKER,
    PARTIAL_MARKER,
    known_extra_state,
    partial_clone_state,
)

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

# Issue 842: the sweeps' population arrives as CONTRACT names, and the repos
# open under the on-disk spellings. The seam below is the one copy of that
# resolution; importing the codec it delegates to.
import repo_alias  # noqa: E402
import tracked_walk  # noqa: E402

console_safe.apply()


_N_ASSERTIONS = 0


def pin_row_exempt(name: str) -> bool:
    """Does this repo owe a pin row? (Issue 821)

    Issue 815's marker landed in `population_verdict` — the FINAL line — and
    not in the per-repo pin loop every sweep runs six hundred lines earlier.
    Measured 2026-09-17 with both documented markers set: **eight of nine
    sweeps red on repos where they found nothing**, each printing "UNPINNED —
    add a row" above a final line reading "not measured and not expected to
    be". Two live ratchet breaches in riir-train were sitting behind those
    reds, which is Issue 793's finding recurring with a second marker.

    ⛔ The caller must pass a repo it actually VISITED. The loop iterates the
    DERIVED walk, so a visited repo is on the box by construction and the
    acknowledged-vs-stale split does not arise here — that split is
    `population_verdict`'s, on the final line, and re-deriving it per row
    would be a second copy of the one rule that makes this marker safe.

    ⚠ NAMES, never `=1` — the asymmetry with the partial marker is 815's whole
    design, and it is what keeps this predicate from becoming a blanket
    excuse: an unnamed extra still owes a row and still reds.

    It DELEGATES to `known_extra_state` rather than reading the marker itself,
    so the one rule that stops this rotting into a blanket excuse exists once:
    a name `repo_set.txt` has SINCE REGISTERED lands in that function's stale
    half, not its acknowledged half, and the repo owes a row again. A private
    `name in declared` test here would have silently skipped that.
    """
    return bool(known_extra_state([name])[0])


def open_repo(name: str, workspace: Path) -> Path:
    """The on-disk directory for a contract repo name (Issue 842).

    Every sweep opens repos through this, never `WORKSPACE / name` directly:
    on a box whose `repo_alias.local.txt` maps on-disk sibling names into the
    contract vocabulary, the contract spelling IS not a directory, and a sweep
    that opens it measures zeros against pins typed from the real repos —
    measured 2026-09-18 across seven sweeps, every red a TRUE pin measured
    against the WRONG DIRECTORY.

    Identity on unaliased boxes (CI, fresh clones, synthetic-workspace
    canaries), and it changes no name the sweep prints: pins, floors and
    verdict lines stay keyed on the CONTRACT spelling — the alias content
    itself must never reach stdout (repo_alias's own rule).
    """
    return Path(workspace) / repo_alias.disk(name)


def zero_walk_floor_accepted(repo: Path) -> tuple[bool, str]:
    """Is a ZERO delegated `min_rs_files` row truthful? (Issue 902)

    Two sweeps (`len_derived_drift_sweep`, `shared_temp_path_drift_sweep`)
    carry no walk floor of their own: they delegate that axis to
    `orphaned_attr_drift_floors.txt` and ASSERT every pinned repo has a
    non-zero row there — a repo whose row is dropped or zeroed has no
    blindness detector at all. A repo born md-only (riir-instinct) has a
    TRUTHFUL zero row, and the assertion read it as a finding forever — a
    sweep that always reds on a correct repo, the cries-wolf state Issue 793
    forbids.

    Measured, never declared: the same instrument the delegated column
    describes (`tracked_walk.tracked_files`, the ONE walk, Issue 777) counts
    the repo's tracked `*.rs`, and the zero row is accepted only while that
    count is zero. The moment a `.rs` file lands — staged (the index counts;
    that is the first half of landing), or merely present on a non-repo tree
    (the fallback counts) — the row MUST be raised, and this returns False so
    the sweeps red exactly as they did before. A zero row is therefore never
    a standing amnesty; it is a measured statement about THIS checkout,
    re-measured every run. (The measured exception is also how the predicate
    caught its first stale premise: riir-reflexer was registered md-only
    2026-09-25 and its row was truthful that day, but its vessel workspace
    had landed 16 tracked `.rs` by the time this shipped — the predicate
    refused the zero row and the rows were re-pinned in the same commit.)

    Returns `(accepted, note)`; the caller prints the note either way, so the
    acceptance is visible on a GREEN run, never inferred from a missing flag.
    """
    files, _excluded = tracked_walk.tracked_files(repo, "*.rs")
    n = len(files)
    if n == 0:
        return True, ("0 tracked .rs at this checkout — md-only repo, the "
                      "zero row is truthful (measured by tracked_walk, not "
                      "declared)")
    return False, f"{n} tracked .rs at this checkout"


def population_verdict(pins, present) -> tuple[list[str], list[str], int]:
    """(lines, deferred, failures) for the rows this run could not measure.

    `pins`     — the repo names this sweep has a row for.
    `present`  — the repo names the derived walk actually found.

    `lines` print immediately (they are findings). `deferred` ride the sweep's
    FINAL line in both directions — a deferral printed only on failure is a
    deferral nobody reads on the run that passes.
    """
    present = sorted(set(present))
    lines: list[str] = []
    deferred: list[str] = []
    failures = 0

    marker_on, snap_absent, unregistered = partial_clone_state(present)

    # Issue 815: repos named as OUTSIDE the contract. `partial_clone_state`
    # has already removed them from `unregistered`; they are disclosed here so
    # the suppression is visible on a PASSING run, not inferred from a bucket
    # that got quieter. A stale acknowledgement is a FINDING, not a deferral —
    # it is the direction in which this marker could rot into a blanket excuse.
    acknowledged, stale_extra = known_extra_state(present)
    if acknowledged:
        deferred.append(
            f"{len(acknowledged)} known-extra repo(s) outside the contract "
            f"({', '.join(acknowledged)}) — acknowledged by "
            f"{KNOWN_EXTRA_MARKER}, not measured and not expected to be")
    if stale_extra:
        lines.append(
            f"⛔ STALE {KNOWN_EXTRA_MARKER} entry (named, but not "
            "unregistered-and-present on this box — gone, or since registered "
            "in repo_set.txt; drop it from the marker): "
            + ", ".join(stale_extra))
        failures += len(stale_extra)

    if unregistered:
        lines.append(
            "⛔ UNREGISTERED (on this box, absent from repo_set.txt — regenerate "
            "it on the canonical workstation and commit): "
            + ", ".join(unregistered))
        failures += len(unregistered)

    # The snapshot widens the set deliberately: a repo in `repo_set.txt` that
    # this sweep has no row for is still a repo this run did not measure, and
    # the per-sweep UNPINNED check cannot see it — that check only fires on
    # repos the walk FOUND.
    absent = sorted((set(pins) | set(snap_absent)) - set(present))
    if absent:
        if marker_on:
            deferred.append(
                f"{len(absent)} contract repo(s) not on this box "
                f"({', '.join(absent)}) — DEFERRED by {PARTIAL_MARKER}=1, "
                "NOT measured by this run")
        else:
            lines.append(
                "⛔ UNSEEN (a contract repo this run could not measure — never a "
                f"pass; set {PARTIAL_MARKER}=1 on a box you KNOW carries a "
                "subset, or remove the row if the repo is genuinely gone): "
                + ", ".join(absent))
            failures += len(absent)

    return lines, deferred, failures


def selftest() -> list[str]:
    """Both postures, and the arm that must red under BOTH.

    Written against the real `repo_set.txt`, because the helper's whole job is
    to compare against that file and a stubbed snapshot would test the stub.
    The arms therefore assert RELATIVE movement (a repo removed from `present`
    becomes absent; a fabricated name becomes unregistered) rather than exact
    counts, which depend on the box.
    """
    import os

    import repo_alias
    from skill_repo_set_gate import SNAPSHOT

    global _N_ASSERTIONS
    _N_ASSERTIONS = 0
    fails: list[str] = []

    def check(cond, msg):
        # ⛔ COUNTED, never typed. The pass line used to hand-type "7
        # assertion(s)"; that is the shape Issue 798 T3 found stale on arrival
        # in `worktree_state`, in a module whose whole subject is records
        # drifting from what they describe. Adding the Issue-815 arms below
        # would have made it wrong again.
        global _N_ASSERTIONS
        _N_ASSERTIONS += 1
        if not cond:
            fails.append(msg)

    if not SNAPSHOT.is_file():
        return ["repo_set.txt is missing — the helper cannot be self-tested"]
    snap = [l.strip() for l in SNAPSHOT.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    check(len(snap) >= 10, f"repo_set.txt reads {len(snap)} repos — too few to "
                           "exercise the arms; the parse is broken")

    saved = os.environ.get(PARTIAL_MARKER)
    saved_extra = os.environ.get(KNOWN_EXTRA_MARKER)
    try:
        # ── no marker ──────────────────────────────────────────────────────
        # BOTH cleared: these arms build a synthetic population but read the
        # real environment, so an ambient marker leaks into arms that predate
        # it — measured in `skill_repo_set_gate`, where exactly that turned a
        # CORRECT invocation into an INSTRUMENT-unreadable verdict (Issue 815).
        os.environ.pop(PARTIAL_MARKER, None)
        os.environ.pop(KNOWN_EXTRA_MARKER, None)
        lines, deferred, n = population_verdict(snap, snap)
        check((lines, deferred, n) == ([], [], 0),
              f"a complete population was not clean: {lines} {deferred} {n}")

        lines, deferred, n = population_verdict(snap, snap[:-1])
        check(n == 1 and deferred == [] and any("UNSEEN" in l for l in lines),
              f"a missing repo without the marker must be UNSEEN: {lines} {n}")

        # ── marker on ──────────────────────────────────────────────────────
        os.environ[PARTIAL_MARKER] = "1"
        lines, deferred, n = population_verdict(snap, snap[:-1])
        check(n == 0 and len(deferred) == 1 and lines == [],
              f"a missing repo WITH the marker must defer, not red: {lines} {n}")
        check(snap[-1] in deferred[0] and PARTIAL_MARKER in deferred[0],
              f"the deferral names neither the repo nor the marker: {deferred}")

        # UNREGISTERED reds under the marker too — this is the arm that proves
        # the marker is not a blanket amnesty.
        lines, deferred, n = population_verdict(snap, snap + ["a-repo-that-joined"])
        check(n == 1 and any("UNREGISTERED" in l for l in lines),
              f"an unregistered repo did not red under the marker: {lines} {n}")

        # A sweep with NO row for a repo the snapshot knows is still missing a
        # measurement — the per-sweep UNPINNED check cannot see it, because
        # that check only fires on repos the walk found.
        lines, deferred, n = population_verdict([], snap[:-1])
        check(len(deferred) == 1 and snap[-1] in deferred[0],
              f"the snapshot did not widen an empty pin set: {deferred}")

        # ── the known-extra axis (Issue 815) ───────────────────────────────
        # The only thing that can take a repo OUT of the UNREGISTERED bucket,
        # so the arms ask what it still reds on.
        os.environ.pop(PARTIAL_MARKER, None)
        joined = snap + ["seal-x", "seal-y"]

        os.environ.pop(KNOWN_EXTRA_MARKER, None)
        lines, deferred, n = population_verdict(snap, joined)
        check(n == 2 and any("UNREGISTERED" in l for l in lines),
              f"two extra repos must both be UNREGISTERED unmarked: {lines} {n}")

        os.environ[KNOWN_EXTRA_MARKER] = "seal-x,seal-y"
        lines, deferred, n = population_verdict(snap, joined)
        check(n == 0 and lines == [],
              f"named known-extra repos must not red: {lines} {n}")
        # Disclosed on the FINAL line of a PASSING run — a suppression nobody
        # sees is a suppression nobody re-reads.
        check(any(KNOWN_EXTRA_MARKER in d for d in deferred),
              f"the acknowledgement is not disclosed: {deferred}")

        # ⚑ The reason the marker takes NAMES: acknowledging one extra repo
        # must not acknowledge the next one. An arm that named both would pass
        # against a blanket `=1` marker too.
        os.environ[KNOWN_EXTRA_MARKER] = "seal-x"
        lines, deferred, n = population_verdict(snap, joined)
        check(n == 1 and any("UNREGISTERED" in l and "seal-y" in l
                             for l in lines),
              f"an UNNAMED extra repo must still red beside a named one: "
              f"{lines} {n}")

        # Both directions: a name that describes nothing is a FINDING, not a
        # deferral — the direction in which this marker rots into an amnesty.
        os.environ[KNOWN_EXTRA_MARKER] = "seal-never-existed"
        lines, deferred, n = population_verdict(snap, snap)
        check(n == 1 and any("STALE" in l for l in lines),
              f"a stale acknowledgement must red: {lines} {n}")
        os.environ[KNOWN_EXTRA_MARKER] = snap[0]
        lines, deferred, n = population_verdict(snap, snap)
        check(n == 1 and any("STALE" in l for l in lines),
              f"acknowledging a REGISTERED repo must red: {lines} {n}")

        # ── pin_row_exempt (Issue 821). The per-repo half of the same marker.
        # Every arm below has a counterpart above, deliberately: the two halves
        # ran on different rules for two days and the disagreement was the
        # defect — the final line said "not expected to be measured" while the
        # rows above demanded a pin so that they could be.
        os.environ[KNOWN_EXTRA_MARKER] = "seal-x,seal-y"
        check(pin_row_exempt("seal-x"),
              "a DECLARED known-extra must not owe a pin row")
        # ⚑ The same NAMES-not-`=1` property, one instrument over. Without this
        # arm the predicate could degrade to "is the marker set at all" and
        # every arm above would still pass.
        check(not pin_row_exempt("seal-unnamed"),
              "an UNDECLARED extra repo must still owe a pin row")
        # The direction in which this rots: a name repo_set.txt has SINCE
        # registered is STALE, and a stale acknowledgement excuses nothing —
        # so the marker cannot only ever loosen.
        os.environ[KNOWN_EXTRA_MARKER] = snap[0]
        check(not pin_row_exempt(snap[0]),
              "acknowledging a REGISTERED repo must not excuse its pin row")
        os.environ.pop(KNOWN_EXTRA_MARKER, None)
        check(not pin_row_exempt("seal-x"),
              "with NO marker set, nobody is excused")

        # ── open_repo (Issue 842) ──────────────────────────────────────────
        # Injected codec state, not the machine's own alias file: the arms
        # must be box-independent, and the file is machine-local (possibly
        # absent). Setting `_loaded` directly is the same cache every accessor
        # reads; `None` restores the lazy re-read of the real file.
        saved_loaded = repo_alias._loaded
        try:
            repo_alias._loaded = {"seal-alias-x": "contract-alias-x"}
            ws = Path("/synthetic-workspace-842")   # pure path math, no I/O
            check(open_repo("contract-alias-x", ws) == ws / "seal-alias-x",
                  "open_repo did not resolve a mapped contract name to its "
                  "on-disk directory")
            check(open_repo("unmapped-name", ws) == ws / "unmapped-name",
                  "open_repo must be identity for an unmapped name")
            # The full codec round-trip the sweeps rely on: the derived walk
            # speaks on-disk, apply() translates to contract for the pins,
            # disk() must take it back.
            check(repo_alias.disk(repo_alias.apply(["seal-alias-x"])[0])
                  == "seal-alias-x",
                  "disk(apply(name)) is not the identity — the sweep's pin "
                  "vocabulary and its open paths would diverge")
            check(repo_alias.display("seal-alias-x") == "contract-alias-x",
                  "display() must return the contract spelling for gate "
                  "output — the alias content must never leak to stdout")
        finally:
            repo_alias._loaded = saved_loaded

        # ── zero_walk_floor_accepted (Issue 902) ───────────────────────────
        # Synthetic trees, because the predicate's whole job is MEASURING a
        # repo, and a stub would test the stub. Both branches of the ONE walk
        # are exercised: the tracked (index) branch on a git repo, and the
        # fallback over a plain tree. The non-Rust arm is also what keeps the
        # first arm honest — a walk that counted every file would have
        # refused md-only too.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            md_only = t / "md-only"
            md_only.mkdir()
            (md_only / "README.md").write_text("# md only\n", encoding="utf-8")
            ok, note = zero_walk_floor_accepted(md_only)
            check(ok and "0 tracked .rs" in note,
                  f"an md-only tree must be accepted: {ok} {note!r}")

            gains_rust = t / "gains-rust"
            (gains_rust / "src").mkdir(parents=True)
            (gains_rust / "src" / "lib.rs").write_text("fn f() {}\n",
                                                       encoding="utf-8")
            ok, note = zero_walk_floor_accepted(gains_rust)
            check(not ok and "1 tracked .rs" in note,
                  f"a tree with .rs must refuse its zero row: {ok} {note!r}")

            docs_only = t / "docs-only"
            docs_only.mkdir()
            (docs_only / "README.md").write_text("# md\n", encoding="utf-8")
            (docs_only / "spec.lean").write_text(
                "theorem t : True := trivial\n", encoding="utf-8")
            ok, note = zero_walk_floor_accepted(docs_only)
            check(ok, f"non-Rust files must not refuse the zero row: "
                      f"{ok} {note!r}")

            # The tracked branch: a STAGED .rs in a git repo refuses even
            # though nothing is committed — `git ls-files` reads the index,
            # and staging is the first half of landing.
            staged = t / "staged-rs"
            staged.mkdir()
            subprocess.run(["git", "-C", str(staged), "init", "-q"],
                           check=True, capture_output=True)
            (staged / "lib.rs").write_text("fn f() {}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(staged), "add", "lib.rs"],
                           check=True, capture_output=True)
            ok, note = zero_walk_floor_accepted(staged)
            check(not ok,
                  f"a STAGED .rs must refuse the zero row: {ok} {note!r}")

            # And the git-repo md-only shape itself: a repo whose index is
            # empty except docs takes the tracked branch and is accepted.
            md_repo = t / "md-repo"
            md_repo.mkdir()
            subprocess.run(["git", "-C", str(md_repo), "init", "-q"],
                           check=True, capture_output=True)
            (md_repo / "BOUNDARY.md").write_text("# contract\n",
                                                 encoding="utf-8")
            subprocess.run(["git", "-C", str(md_repo), "add", "BOUNDARY.md"],
                           check=True, capture_output=True)
            ok, note = zero_walk_floor_accepted(md_repo)
            check(ok and "0 tracked .rs" in note,
                  f"a git repo with no tracked .rs must be accepted: "
                  f"{ok} {note!r}")
    finally:
        if saved is None:
            os.environ.pop(PARTIAL_MARKER, None)
        else:
            os.environ[PARTIAL_MARKER] = saved
        if saved_extra is None:
            os.environ.pop(KNOWN_EXTRA_MARKER, None)
        else:
            os.environ[KNOWN_EXTRA_MARKER] = saved_extra

    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("sweep_population selftest FAILED:")
        for f in fails:
            print("  ✗ " + f)
        return 1
    print(f"✓ sweep_population selftest — {_N_ASSERTIONS} assertion(s), "
          "COUNTED not typed: complete population clean, UNSEEN without the "
          "marker, DEFERRED with it (naming repo + marker), UNREGISTERED reds "
          "under the marker, snapshot widens an empty pin set, the "
          f"{KNOWN_EXTRA_MARKER} axis both ways (named extras excused and "
          "disclosed, unnamed ones still red, stale entries red), and the "
          "Issue-902 zero-walk-floor predicate (md-only accepted both walk "
          "branches, non-Rust ignored, staged and present .rs refuse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
