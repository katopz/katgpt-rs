#!/usr/bin/env python3
"""Issue 513 T4 — build the `required-features` rows a push actually TOUCHES.

The full sweep (`required_features_build_audit.py`) is the right instrument and
the wrong cadence: 1,866 rows over 16 repos at a ~28 s/row mean is hours per
large repo, so it is a scheduled job. The defect it finds, though, arrives in a
*commit* — every instance so far was a target that was copied, edited, or
declared in the same push that broke it:

  | instance | repo | shape |
  |---|---|---|
  | `test_cubecl_backward_grads` | riir-train | copy of a twin, lost a feature |
  | `bench_001_pruners_goat`     | katgpt-rs  | copy of a twin, lost a feature |
  | `transition_error_taxonomy_battery` | riir-neuron-db | an over-claiming cfg on an import |
  | `dasd_lora_goat`             | riir-train | row omitted a second feature |
  | `goat_235b_filter_training`  | riir-train | ungated import, cfg'd use site |

So the affordable gate is not "check every row", it is "check the rows this
push could have broken". Cost is bounded by the diff, not by the repo.

WHAT IT SELECTS (union of two, both cheap and both sound):

  1. a changed file that IS a row's target source  -> that row
  2. a changed Cargo.toml                          -> the rows whose OWN
     (kind, name, required-features) tuple differs between base and head

Rule 2 is a row diff, NOT "every row in the package", and the difference is
what makes the gate affordable at all: `riir-train-gpu`'s manifest carries 440
rows, so selecting the package would refuse every manifest edit in the repo
where three of the five instances live. Both sides are parsed by the sweep's
own `rows_from_manifest`, so the per-push gate and the scheduled sweep cannot
disagree about what a row IS.

WHAT IT DELIBERATELY DOES NOT SELECT, and why a green must be read narrowly:
a changed `src/**.rs` can break a row in any dependent package — that is how
instance 1 happened, a re-export gate widened in the library. Selecting on it
is unbounded (touching `katgpt-core/src/lib.rs` selects the workspace), so it
is left to the sweep and REPORTED as unchecked rather than silently omitted.
`--src-fanout` opts into it, capped by `--max-rows`, for a push whose diff is
small enough to afford it.

Exit codes:
  0  every selected row builds at exactly its own feature set (or none selected)
  1  a selected row FAILS-TO-BUILD / NO-SUCH-FEATURE / TIMEOUT / UNSEEN
  2  the instrument is untrustworthy (selftest failed)

UNSEEN is exit 1, not exit 0. A row with neither an error nor an artifact is
undecided, and this family's standing rule is that silence is not evidence — a
gate that reports an undecided row as green is the green zero it exists to
refuse. (A `target_os`-gated target does NOT land here: cargo still emits an
artifact for the empty test target, so it reports BUILDS.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from required_features_build_audit import (  # noqa: E402
    BUILDS,
    DiskFull,
    Row,
    check_group,
    disk_headroom_ok,
    free_gib,
    print_reclaim_hint,
    group_rows,
    parse_rows,
    rows_from_manifest,
    MIN_FREE_GIB,
)


def changed_files(repo: Path, base: str, head: str) -> list[Path]:
    """Paths changed between two revisions, resolved absolute.

    `--diff-filter=d` drops DELETED files: a removed target source cannot be
    built and its row is either gone with it (nothing to check) or orphaned
    (a different defect, owned by the manifest audits).
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", "--diff-filter=d", base, head],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git diff {base}..{head} failed: {proc.stderr.strip()}")
    return [(repo / ln).resolve() for ln in proc.stdout.splitlines() if ln.strip()]


def show(repo: Path, rev: str, rel: str) -> str | None:
    """A file's content at a revision, or None if it did not exist there.

    None and empty are kept apart on purpose: a manifest ADDED by this push has
    no base version, and every row in it is new — which is exactly the
    introduce-commit this gate exists to catch, not a reason to select nothing.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{rel}"],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def row_key(r: Row) -> tuple[str, str, tuple[str, ...]]:
    """What makes a row the same row for diffing purposes.

    Features are a SET, not a list: reordering `["a","b"]` to `["b","a"]` is not
    a change and must not select the row. Whether it is `[[test]]` or `[[bench]]`
    IS part of the identity — cargo builds them differently, and two targets can
    legitimately share a name across kinds.
    """
    return (r.kind, r.name, tuple(sorted(set(r.features))))


def changed_rows(repo: Path, manifest: Path, base: str, head_rows: list[Row]) -> list[Row]:
    """Rows in this manifest that are new or whose feature set moved.

    A row REMOVED between base and head yields nothing: there is no target left
    to build. A row whose features shrank yields the head row, which is the
    single commonest shape in the record (a copy that lost a feature).
    """
    rel = str(manifest.resolve().relative_to(repo))
    mine = [r for r in head_rows if Path(r.crate_dir) == manifest.parent.resolve()]
    text = show(repo, base, rel)
    if text is None:
        return mine  # manifest is new in this push — every row in it is new
    before = {row_key(r) for r in rows_from_manifest(text, repo.name, manifest)}
    return [r for r in mine if row_key(r) not in before]


def select(
    rows: list[Row],
    changed: list[Path],
    src_fanout: bool,
    changed_rows_for: "Callable[[Path], list[Row]]",
) -> tuple[list[Row], list[str]]:
    """The rows a change set touches, plus the reason each was selected.

    `changed_rows_for` is injected rather than called directly so the selftest
    can pin the selection rules without a git repository — the two failure
    modes this gate has (a selector too narrow to fire, one too wide to afford)
    both live in this function, and a test that needs a fixture repo is a test
    nobody runs.
    """
    by_path = {r.path: r for r in rows}
    picked: dict[str, Row] = {}
    why: list[str] = []

    for f in changed:
        row = by_path.get(str(f))
        if row is not None:
            picked.setdefault(row.label, row)
            why.append(f"target-source  {row.label}")
            continue
        if f.name == "Cargo.toml":
            for r in changed_rows_for(f):
                if r.label not in picked:
                    picked[r.label] = r
                    why.append(f"row-changed    {r.label}")

    if src_fanout:
        # Every row in a package whose own `src/` changed. NOT transitive:
        # a dependent package's rows are still the sweep's job. Bounded by
        # --max-rows at the call site, never here — a silent truncation would
        # be the blind spot this whole module is written to name out loud.
        src_pkgs: set[str] = set()
        for f in changed:
            if f.suffix != ".rs":
                continue
            for r in rows:
                if str(f).startswith(r.crate_dir + os.sep + "src" + os.sep):
                    src_pkgs.add(r.package)
        for pkg in sorted(src_pkgs):
            for r in rows:
                if r.package == pkg and r.label not in picked:
                    picked[r.label] = r
                    why.append(f"src({pkg})      {r.label}")

    return list(picked.values()), why


def selftest() -> None:
    """Pin selection in BOTH directions — a narrow selector is a silent pass.

    Exits 2, never 1: an untrustworthy instrument is a different finding from a
    real defect, and collapsing them is how a broken classifier reads as a clean
    repo (the lesson `percentile_floor_gate.py` carries).
    """

    def row(pkg: str, name: str, path: str, crate: str, feats: list[str]) -> Row:
        return Row(
            repo="r", package=pkg, kind="test", name=name,
            features=feats, path=path, crate_dir=crate,
        )

    a = row("pkg_a", "t_a", "/w/a/tests/t_a.rs", "/w/a", ["f"])
    b = row("pkg_a", "t_b", "/w/a/tests/t_b.rs", "/w/a", ["f", "g"])
    c = row("pkg_b", "t_c", "/w/b/tests/t_c.rs", "/w/b", ["h"])
    rows = [a, b, c]
    none: Callable[[Path], list[Row]] = lambda _p: []
    fails: list[str] = []

    def want(got: list[Row], exp: set[str], case: str) -> None:
        have = {r.name for r in got}
        if have != exp:
            fails.append(f"{case}: selected {sorted(have)}, want {sorted(exp)}")

    # 1. a touched target source selects exactly its own row — not its neighbour.
    want(select(rows, [Path("/w/a/tests/t_a.rs")], False, none)[0], {"t_a"}, "target-source")
    # 2. a touched manifest selects what the row diff returns, and ONLY that.
    #    Selecting the whole package instead would refuse every manifest edit in
    #    riir-train-gpu (440 rows) — the repo three of the five instances are in.
    want(select(rows, [Path("/w/a/Cargo.toml")], False, lambda _p: [b])[0], {"t_b"}, "row-changed")
    # 3. a manifest whose rows did NOT move selects nothing. Without this the
    #    gate degrades to "check the package" and stops being affordable.
    want(select(rows, [Path("/w/a/Cargo.toml")], False, none)[0], set(), "manifest-nodiff")
    # 4. an unrelated file selects nothing. Without this the gate is vacuous in
    #    the common case (most pushes touch no target and no manifest) and a
    #    green would mean "the selector is broken", indistinguishable.
    want(select(rows, [Path("/w/a/README.md")], False, none)[0], set(), "unrelated")
    # 5. src fanout is OFF by default — the whole cost argument depends on it.
    want(select(rows, [Path("/w/a/src/lib.rs")], False, none)[0], set(), "src-off")
    # 6. ...and ON it selects that package only, never the workspace.
    want(select(rows, [Path("/w/a/src/lib.rs")], True, none)[0], {"t_a", "t_b"}, "src-on")
    # 7. a path that merely PREFIXES a crate dir is not inside it. `/w/ab/src`
    #    starts with `/w/a` as a string; the separator is what makes it a
    #    directory test. Same shape as the `.git`-is-a-file case in
    #    population_sync_gate.py.
    d = row("pkg_ab", "t_d", "/w/ab/tests/t_d.rs", "/w/ab", ["f"])
    want(select(rows + [d], [Path("/w/ab/src/lib.rs")], True, none)[0], {"t_d"}, "prefix")
    # 8. dedupe: a push touching a target AND moving its row must not build it
    #    twice, and must not report it twice either.
    got, wh = select(rows, [Path("/w/a/tests/t_a.rs"), Path("/w/a/Cargo.toml")],
                     False, lambda _p: [a])
    if len(got) != 1 or len(wh) != 1:
        fails.append(f"dedupe: {len(got)} rows / {len(wh)} reasons, want 1 / 1")

    # ── row_key: what counts as "the row moved" ──
    if row_key(a) == row_key(row("pkg_a", "t_a", a.path, "/w/a", ["f", "g"])):
        fails.append("row_key: a feature ADDED to a row did not register as a change")
    if row_key(b) != row_key(row("pkg_a", "t_b", b.path, "/w/a", ["g", "f"])):
        fails.append("row_key: reordering a feature list registered as a change")
    if row_key(b) != row_key(row("pkg_a", "t_b", b.path, "/w/a", ["f", "g", "f"])):
        fails.append("row_key: a duplicated feature registered as a change")
    if row_key(a) == row_key(Row(repo="r", package="pkg_a", kind="bench", name="t_a",
                                 features=["f"], path=a.path, crate_dir="/w/a")):
        fails.append("row_key: [[test]] and [[bench]] of one name collapsed")

    if fails:
        for f in fails:
            print(f"selftest FAIL: {f}", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", nargs="?", default=".", help="repo to gate")
    ap.add_argument("--base", default="HEAD~1", help="diff base (default HEAD~1)")
    ap.add_argument(
        "--head",
        default="HEAD",
        help="diff head (default HEAD). NOTE it selects the changed FILE LIST "
        "only — a row's current definition always comes from the CHECKOUT, "
        "because that is what would be built. The two coincide in the case "
        "that matters (CI, where the checkout IS head); passing an older "
        "--head widens the window rather than narrowing it, which is the safe "
        "direction but is not what the flag's name suggests",
    )
    ap.add_argument("--src-fanout", action="store_true",
                    help="also select rows in a package whose own src/ changed")
    ap.add_argument("--max-rows", type=int, default=24,
                    help="refuse (exit 1) above this many selected rows; a push "
                         "that wide belongs to the sweep, and silently "
                         "truncating would be the blind spot this gate names")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--target-dir", default=None)
    ap.add_argument("--list", action="store_true", help="select only, no builds")
    args = ap.parse_args(argv)

    selftest()

    repo = Path(args.repo).resolve()
    rows = parse_rows(repo)
    changed = changed_files(repo, args.base, args.head)
    picked, why = select(
        rows,
        changed,
        args.src_fanout,
        lambda m: changed_rows(repo, m, args.base, rows),
    )

    print(f"{repo.name}: {len(changed)} changed files, {len(rows)} rows in repo, "
          f"{len(picked)} selected (src-fanout {'on' if args.src_fanout else 'OFF'})")
    for line in why:
        print(f"  select  {line}")
    if not args.src_fanout:
        print("  NOTE: rows reachable only through a library change are NOT "
              "checked here — that class is the sweep's. Read a green narrowly.")
    if not picked:
        return 0
    if len(picked) > args.max_rows:
        print(f"REFUSE: {len(picked)} selected > --max-rows {args.max_rows}. "
              f"Run required_features_build_audit.py over this package instead.")
        return 1
    if args.list:
        return 0

    # A full disk makes every verdict below meaningless, and the way it presents
    # is the reason this check is a REFUSE and not a warning: cargo cannot write
    # the artifact, emits neither a compiler-artifact nor a compiler-message,
    # and the row reports UNSEEN — which this gate treats as exit 1, i.e. as a
    # WRONG ROW. Measured on a 515-row riir-ai sweep: 26 rows read as UNSEEN
    # with 2.5 GiB free, and all 26 built on a re-run with room. Exit 2, because
    # an untrustworthy instrument is not the same finding as a bad row.
    if not disk_headroom_ok(args.target_dir):
        print(f"REFUSE: {free_gib(args.target_dir):.1f} GiB free on the "
              f"filesystem holding the target dir, below the {MIN_FREE_GIB} GiB "
              f"floor. Rows would report UNSEEN for an environmental reason and "
              f"this gate would red them as wrong rows. Free space, pass "
              f"--target-dir somewhere with room, or set RFBA_MIN_FREE_GIB.",
              file=sys.stderr)
        print_reclaim_hint(str(repo))
        return 2

    bad: list[tuple[str, str, str]] = []
    for group in group_rows(picked):
        try:
            results = check_group(repo, group, args.target_dir, args.timeout)
        except DiskFull as e:
            print(f"\nABORT: {e}\n  {free_gib(args.target_dir):.1f} GiB free. "
                  f"Rows checked before this point are valid; the rest are not "
                  f"decided, and reporting them as failures would be wrong.",
                  file=sys.stderr)
            return 2
        for res in results:
            mark = "  " if res.verdict == BUILDS else "!!"
            print(f"{mark} {res.verdict:<16} {res.seconds:5.1f}s {res.row.label}"
                  + (f"  ({res.detail})" if res.detail else ""))
            if res.verdict != BUILDS:
                bad.append((res.row.label, res.verdict, res.detail))

    if bad:
        print(f"\nFAIL: {len(bad)} of {len(picked)} touched rows do not build at "
              f"their own feature set.")
        for label, verdict, detail in bad:
            print(f"  {verdict:<16} {label}  {detail}")
        print("\nA row that exists and is wrong reads as PROTECTED in every "
              "audit: `cargo test --workspace` skips the target, --all-features "
              "builds it, and the 'w/ req-f' column counts it. See "
              "riir-train/.issues/513.")
        return 1
    print(f"\nok — {len(picked)}/{len(picked)} touched rows build at their own "
          f"feature set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
