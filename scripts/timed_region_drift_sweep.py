#!/usr/bin/env python3
"""The timed-region guard verdict over every contract repo, PINNED (Issue 855 T6).

The per-push half is `timed_region_guard_gate.py`, which gates THIS repo by
MEMBERSHIP over a population Issue 855 T3 executed end to end — 34 regions run,
every pinned row carrying the number somebody MEASURED. That standard cannot be
exported, and T5's own close-out says why:

> a wall in a sibling would be a wall over an unread bucket, which is exactly
> what T4 refused.

⛔ So this is a **RATCHET**, and the ratchet is EARNED rather than assumed.
T5 executed all 33 sibling READ-tier rows (2026-09-19) — 1 UNBUILDABLE, 32 run,
**2 VANISHED (6.3%)** against this repo's own 7 of 34 (20.6%). That is enough
evidence to constrain the DERIVATIVE in somebody else's tree: the commit that
adds ANOTHER unguarded timed region reds, and the rows already there stay their
owners' to adjudicate. It is NOT enough for a membership wall with a reason per
row — 30 of the 32 are single readings taken on a loaded box without T3's 4x
scaling probe, so *fast* and *partly eliminated* are not separated for them.
That residue is stated in Issue 855 and not laundered into a pin here.

⚠ Issue 785's rule is respected rather than skirted. It forbids ratcheting a
bucket whose meaning is *unanswered*; this bucket means *no loud-zero defence*,
every row has an owner, and the population behind the number has been RUN.

Three states it reports and does not gate — all three found by EXECUTING
--------------------------------------------------------------------------
None is foldable into the guard verdict, and each was invisible to the counting
half of T5:

- **`#[ignore]`d.** `cargo test --exact <fn>` prints `ok. 0 passed; 1 ignored`
  and exits **0** — byte-for-byte this family's own green-zero shape, met
  inside T5's own runner and nearly recorded as a result. The region is then
  neither guarded nor satisfied by absent work; it is UNEXECUTED. Measured: all
  4 riir-chain rows in T5's set, live once re-run with `--ignored`.
- **prints NO number.** The quantity lives only inside an `assert!` message,
  i.e. it is visible ONLY on failure — so *read the printed value next to the
  bar*, the method that found every other row, cannot see it at all. Measured
  as the highest-yield slice by a factor of four: 1 VANISHED of 4 (25%) against
  a population rate of 6.3%.
- **UNBUILDABLE** is NOT reported, because it is not statically decidable and
  this sweep would have to guess. riir-chain's workspace-excluded bridge crate
  cannot resolve its dependencies at all, so its GOAT gate is compiled by
  nothing — a manifest-RESOLUTION property, filed there as `.issues/157`. Named
  here so a later census reads the exclusion instead of re-deriving it.

Two floors, and in 8 of 21 repos NEITHER bites
-----------------------------------------------
`min_files` catches the walk going blind; `min_regions` catches the PREDICATE
going blind over an unchanged walk. They break separately — a `git ls-files`
regression takes the first to 0 with the classifier intact, and a masker or
regex regression takes the second to 0 while the walk is untouched.

⚠ Eight repos have **0 timed regions** and three have **0 tracked test/bench
files**, so both quantities are 0 there and neither detects anything — Issue
783's population shape, stated as a measurement rather than assumed. A reserved
`TOTALS` row floors the population GLOBALLY for exactly that reason, which is
the `wasm32_surface_drift_sweep` answer to the same problem.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical to the other sweeps: CI has one checkout, the siblings are simply
absent, so this would either red on every run or derive an EMPTY population and
print a confident green over zero repos.

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy**. `--canary` runs the adversary arms.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import console_safe  # noqa: E402

console_safe.apply()

# DRY: the classifier is the per-push gate's, imported and never restated, so
# the two can never disagree about what "unguarded" means.
import timed_region_guard_gate as trg  # noqa: E402
from skill_repo_set_gate import derive_repos  # noqa: E402
from sweep_population import open_repo, pin_row_exempt, population_verdict  # noqa: E402
from worktree_state import deferral_line, head_delta, sweep_advisory  # noqa: E402

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "timed_region_drift_floors.txt"

FIELDS = ("min_files", "min_regions", "max_unguarded")
TOTALS = "TOTALS"

# Everything that can change a verdict here is the region's OWN file: the
# classifier is per-file (its one cross-region input, the file-scope literal
# binds, is also per-file). That is the premise `head_delta` requires, and it
# is why this sweep may use the cheap per-file shortcut where
# `instrument_reachability` and `len_derived` may not.
SCOPE = ("*.rs",)


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(f"malformed pin row (want {1 + len(FIELDS)} fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(v) for v in parts[1:])))
    return rows


def adjudicate(repo: Path, rows: list[str]):
    """The READ-tier rows, split COMMITTED / UNCOMMITTED / MASKED.

    The DISPLAY reads the worktree; the PINS read `.head`. A row's key is
    `<path>::<fn>` — deliberately LINE-FREE, so an edit above a region does not
    report it as both UNCOMMITTED and MASKED at once.
    """

    def rescan(rel: str, head_src: str | None) -> list[str]:
        # ⚠ `None` means the path is TRACKED but absent from HEAD — a
        # staged-but-never-committed file, Issue 822's measured case. It must
        # answer with no rows rather than crash: such a file cannot carry a
        # row HEAD would report.
        if head_src is None:
            return []
        return [f"{rel}::{n}" for n in trg.classify(head_src).unguarded]

    return head_delta(repo, SCOPE, rows,
                      lambda r: r.split("::", 1)[0],
                      lambda r: r,
                      rescan)


def selftest() -> list[str]:
    fails: list[str] = []

    # The gate's own arms cover the classifier; INVOKING them is what makes
    # "shared classifier" an assertion rather than an import statement.
    try:
        trg.selftest()
    except AssertionError as e:
        fails.append(f"gate selftest: {e}")
    for attr in ("classify", "scan", "MIN_FILES", "MIN_REGIONS", "FileScan"):
        if not hasattr(trg, attr):
            fails.append(f"gate lost `{attr}` — the sweep shares its closure "
                         "and must not fall back to a copy")

    # katgpt-rs's own two floors are the GATE's constants, not this file's
    # opinion — the same quantity in two files is the drift
    # `docs_gate_paths_sync.py` exists for one axis over.
    if PINS.is_file():
        try:
            mine = parse_pins(PINS).get(REPO_ROOT.name, {})
        except ValueError as e:
            fails.append(f"own pins unreadable: {e}")
            mine = {}
        for field, owned in (("min_files", trg.MIN_FILES),
                             ("min_regions", trg.MIN_REGIONS)):
            if mine and mine.get(field) != owned:
                fails.append(
                    f"pin drift: {PINS.name} says {field}={mine.get(field)} for "
                    f"{REPO_ROOT.name}, the gate owns {owned} — change both")
        # The CEILING is the gate's too: its pin file holds one measured row
        # per READ-tier region, so this repo's max_unguarded IS that row
        # count. Unasserted, it drifted for 11 days (49fa7aa17 pinned
        # sync_bank_goat's g2b_readout in the gate; this file stayed at 27).
        if mine and trg.PINS.is_file():
            n_rows = sum(1 for k in trg.read_pins(trg.PINS) if not k.startswith("__"))
            if mine.get("max_unguarded") != n_rows:
                fails.append(
                    f"pin drift: {PINS.name} says max_unguarded="
                    f"{mine.get('max_unguarded')} for {REPO_ROOT.name}, the "
                    f"gate pins {n_rows} row(s) in {trg.PINS.name} — change both")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.txt"
        p.write_text("# c\nrepo-a 40 8 7  # trailing\n\n", encoding="utf-8")
        if parse_pins(p) != {"repo-a": dict(zip(FIELDS, (40, 8, 7)))}:
            fails.append("pin parse: 3-field row not read correctly")
        p.write_text("repo-a 1 2\n", encoding="utf-8")
        try:
            parse_pins(p)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass

    # ⛔ The reserved TOTALS row must NOT be treated as a repo. A sweep that
    # tried to open `WORKSPACE / "TOTALS"` would walk nothing and report a
    # confident zero for the one floor that guards the whole population.
    if TOTALS in derive_repos(WORKSPACE):
        fails.append("a real repo is named TOTALS — the reserved row collides")

    return fails


def main(argv: list[str]) -> int:
    fails = selftest()
    if fails:
        for f in fails:
            print(f"⛔ instrument self-test: {f}")
        return 2
    if "--canary" in argv:
        print(f"✓ timed-region drift sweep — {len(fails)} self-test failure(s)")
        return 0

    try:
        pins = parse_pins(PINS)
    except ValueError as e:
        print(f"⛔ {PINS.name}: {e}")
        return 2
    totals_pin = pins.pop(TOTALS, None)
    if totals_pin is None:
        print(f"⛔ {PINS.name} has no reserved `{TOTALS}` row — with 8 repos at "
              "0 regions the per-repo floors are vacuous there and NOTHING "
              "floors the population")
        return 2

    present = sorted(derive_repos(WORKSPACE))
    lines, deferred, pop_failures = population_verdict(set(pins), present)
    for line in lines:
        print(line)

    rc = pop_failures
    tot = dict.fromkeys(("files", "regions", "unguarded", "ignored", "silent"), 0)
    # `sweep_advisory` is called ONCE for the whole run, with the repos this
    # run actually MEASURED — not per repo, and never re-derived: an advisory
    # about a repo the sweep never read is noise.
    measured: list[Path] = []
    n_uncommitted = n_masked = 0

    for name in present:
        repo = open_repo(name, WORKSPACE)
        if repo is None:
            continue
        sc = trg.scan(repo)
        rows = list(sc.unguarded)
        d = adjudicate(repo, rows)
        head_n = len(d.committed) + len(d.masked)

        measured.append(repo)
        n_uncommitted += len(d.uncommitted)
        n_masked += len(d.masked)
        tot["files"] += sc.files
        tot["regions"] += sc.regions
        tot["unguarded"] += len(rows)
        tot["ignored"] += len(sc.ignored)
        tot["silent"] += len(sc.silent)

        row = pins.get(name)
        if row is None:
            if not pin_row_exempt(name):
                # The measured numbers ride the refusal (the sibling sweeps'
                # shape) — a pin needs them, and a bare UNPINNED forced a
                # second instrument to read them.
                print(f"  ✗ {name}: UNPINNED — add a row to {PINS.name} "
                      f"(measured files={sc.files} regions={sc.regions} "
                      f"READ-unguarded={head_n})")
                rc = 1
            continue

        extra = ""
        if d.uncommitted:
            extra += f", {len(d.uncommitted)} uncommitted"
        if d.masked:
            extra += f", {len(d.masked)} MASKED"
        print(f"  {name:<24} files={sc.files:<5} regions={sc.regions:<5} "
              f"READ-unguarded={head_n}/{row['max_unguarded']:<4} "
              f"#[ignore]d={len(sc.ignored)} silent={len(sc.silent)}{extra}")

        if sc.files < row["min_files"]:
            print(f"      ✗ walk floor — {sc.files} file(s) < {row['min_files']}")
            rc = 1
        if sc.regions < row["min_regions"]:
            print(f"      ✗ predicate floor — {sc.regions} region(s) < "
                  f"{row['min_regions']}; the classifier went blind over an "
                  "unchanged walk")
            rc = 1
        if head_n > row["max_unguarded"]:
            print(f"      ✗ RATCHET — {head_n} > {row['max_unguarded']}: a NEW "
                  "unguarded timed region landed. Give it a loud-zero defence "
                  "(`best_of_us` / `ab_median_ratio`, or an "
                  "`assert!(elapsed.as_nanos() > 0)`), or RUN it, read the "
                  "printed number, and raise the pin with that number as the "
                  "reason.")
            for k in sorted(set(d.committed) | set(d.masked))[:10]:
                print(f"          · {k}")
            rc = 1
        elif head_n < row["max_unguarded"]:
            print(f"      ⚠ {head_n} is BELOW the pin of {row['max_unguarded']} — "
                  "tighten it in the same commit that lowered it, or the slack "
                  "is a free pass for the next one. Not a failure.")

    # The GLOBAL floor, because the per-repo ones are vacuous in 8 of 21.
    if tot["files"] < totals_pin["min_files"] or tot["regions"] < totals_pin["min_regions"]:
        print(f"  ✗ {TOTALS}: population floor — {tot['files']} file(s) / "
              f"{tot['regions']} region(s) against "
              f"{totals_pin['min_files']} / {totals_pin['min_regions']}")
        rc = 1

    print(f"  {'TOTALS':<24} files={tot['files']} regions={tot['regions']} "
          f"READ-unguarded={tot['unguarded']} #[ignore]d={tot['ignored']} "
          f"silent={tot['silent']}")
    # ⛔ `deferral_line`, never a hand-rolled `⚠ {a}`: these lines come from TWO
    # sources and `sweep_advisory`'s already carry their own glyph, so
    # prefixing one produced `⚠ ⚠ STALE:` on the first run of this sweep — the
    # doubled-glyph defect that helper exists to prevent, reproduced here
    # within minutes of it being documented.
    for a in sweep_advisory(measured, SCOPE,
                            uncommitted_rows=n_uncommitted,
                            masked_rows=n_masked):
        print(f"  {deferral_line(a)}")
    # Deferrals ride the FINAL lines in BOTH directions — a deferral printed
    # only on failure is one nobody reads on the run that passes.
    for d in deferred:
        print(f"  {deferral_line(d)}")

    verdict = "✓ timed-region drift sweep PASSED" if rc == 0 else \
              "✗ timed-region drift sweep FAILED"
    print(f"{verdict} — the ratchet constrains what lands NEXT, never what "
          "already landed. ⚠ `#[ignore]d` and `silent` are REPORTED and never "
          "gated: each is orthogonal to the guard, and UNBUILDABLE is not "
          "statically decidable at all (riir-chain Issue 157).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
