#!/usr/bin/env python3
"""Run the trap-sentinel verdicts over EVERY contract repo, not just this one.

`scripts/trap_sentinel_gate.py` is katgpt-rs-scoped by construction — its
`PINNED_SENTINELLED` names two files in this repo and `docs_gate.yml` has a
single checkout, so it could never see a sibling. That is the right shape for
a per-push CI gate and the wrong shape for "is anybody ELSE about to report a
PASS on an abort?"

The workspace answer to that question was just bought and is being held by
nothing. Issue 734 took the population from 38 EXPOSED to **1**, across ten
repos and 37 scripts:

    katgpt-rs             70eff640    2      riir-clippy      c152b80     2
    riir-chain            e3abbb3d   11      riir-dapps       0148fc8     2
    riir-train            4d35aed4    8      riir-auth        bd50158     1
    riir-mmorpg-examples  3f20650     6      riir-deployer    a632a4b     1
    riir-ai               a8260ad23   4      riir-viewbridge  21da73a     1
    seal-remake           26a18191 + 7efe2a23                             2

Every one of those 37 sentinels is a single line that a future edit can drop
without a word from any gate. That is precisely the shape of the defect Issue
734 exists about: seal-remake's guard could not fail past its layer 13 for
months because nothing objected at the time.

This is the fifth instance of one shape in this workspace, and the first two
found real defects the moment they were pointed anywhere but here:

    Issue 702  ci_gate_coverage              one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep         one repo -> 35 duplicate numbers
    2026-09-06 required_features_drift_sweep one repo -> clean, and pinned there
    2026-09-06 percentile_drift_sweep        one repo -> clean, and pinned there
    this file  trap_sentinel_drift_sweep     one repo -> 1 finding, pinned + proven inert

Why the ceilings alone hold the sentinels — no membership pin needed
-------------------------------------------------------------------
`cfg_gated_floor_gate.py` and `trap_sentinel_gate.py` both pin their sets by
MEMBERSHIP, because a count is not a checksum over a set: a swap keeps the
total stable. That argument does **not** apply here, and it is worth being
explicit about why rather than copying the stricter thing by reflex.

There, SENTINELLED was a NAME the pin had to carry. Here the verdict is
DERIVED from the file, and every way to lose a sentinel lands in a class the
ceilings already cover:

    delete the flag from an errexit script   -> EXPOSED        (max_exposed)
    delete the flag from a nounset-only one  -> PRECAUTIONARY  (max_precautionary)
    delete the whole script                  -> population floor
    remove its `set -e` / `set -u`           -> population floor
    remove its EXIT trap                     -> population floor

A membership list of 40 names would add nothing the five pins above do not
already catch, and would have to be re-typed on every legitimate rename.

Why BOTH floors, and why the walk floor is not redundant
--------------------------------------------------------
`max_exposed = 0` is green over whatever the classifier can SEE, so a
regression in `walk_sh` (it shells out to `git ls-files`) takes the
population to 0 and every ceiling passes — indistinguishable from a clean
repo. `min_population` catches that.

It cannot do the job alone: **six of the seventeen repos have a population of
ZERO** (no script with both an abort-on-error option and an EXIT trap), so
their population floor is 0 and detects nothing at all. `min_scripts` — the
size of the tracked-`*.sh` walk that produced the population — still bites
there, and it is the quantity a `git ls-files` regression actually moves.
Same argument as `percentile_drift_sweep.py`'s `min_rs_files`.

Both floors are deliberately SLACK against churn (~60% of measured) and TIGHT
against blindness: consolidating three ad-hoc gate scripts into one
legitimately shrinks the count by a third, and a floor that ratchets to the
last measurement would red that refactor and teach whoever hits it that the
sweep is noise. A walk regression drops these by an order of magnitude.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical to the other four sweeps: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script                 workstation, on demand, every contract repo
    trap_sentinel_gate.py       CI, per-push (docs_gate.sh), katgpt-rs only

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy** — an unreliable instrument is not the same finding as drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the classifier, the premise measurement and the selftest are the
# report's, so the sweep, the per-push gate and the report can never disagree
# about what EXPOSED means.
import trap_exit_launder_audit as tela  # noqa: E402
import trap_sentinel_gate as tsg  # noqa: E402

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "trap_sentinel_drift_floors.txt"

FIELDS = ("min_scripts", "min_population", "max_exposed", "max_precautionary",
          "max_live_forward", "max_unparsed", "max_replaced")
CLASSES = (("exposed", tela.EXPOSED), ("precautionary", tela.PRECAUTIONARY),
           ("live_forward", tela.LIVE_FORWARD), ("unparsed", tela.UNPARSED))


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(
                f"malformed pin row (want {1 + len(FIELDS)} fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(v) for v in parts[1:])))
    return rows


def audit(repo: Path) -> dict:
    """One repo -> the gated classes + BOTH populations that produced them."""
    got = {"n_scripts": 0, "n_pop": 0, "replaced": 0, "rows": []}
    for name, _ in CLASSES:
        got[name] = []
    for path in tela.walk_sh(str(repo)):
        got["n_scripts"] += 1
        r = tela.analyse(path)
        if r is None:
            continue
        got["n_pop"] += 1
        r = dict(r, file=str(Path(path).relative_to(repo)))
        got["rows"].append(r)
        got["replaced"] += bool(r["replaced"])
        for name, verdict in CLASSES:
            if r["verdict"] == verdict:
                got[name].append(r)
    return got


def selftest() -> list[str]:
    """Pin that the verdicts FIRE through THIS sweep's `audit()`, that the
    control does not, that the population derivation holds, and the parser.
    Each fails silently otherwise, and a silent failure reports a clean
    workspace."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = ws / "fake-repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "BOUNDARY.md").write_text("x")
        (repo / ".git").mkdir()

        exposed = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'A="$(mktemp)"\n'
            "trap 'rm -f \"$A\"' EXIT\n"
            'echo "$SOMETHING"\necho ALL GREEN\n'
        )
        (repo / "scripts" / "g.sh").write_text(exposed)
        # `walk_sh` asks git; a temp dir has no index, and its documented
        # fallback is an on-disk walk. Assert we actually got the file, or the
        # whole selftest silently measures an empty population.
        got = audit(repo)
        if got["n_scripts"] != 1:
            fails.append(f"walk found {got['n_scripts']} script(s), expected 1 — "
                         f"the non-git fallback in walk_sh regressed and every "
                         f"selftest arm below is vacuous")
        elif len(got["exposed"]) != 1:
            fails.append(f"planted EXPOSED script: got "
                         f"{[r['verdict'] for r in got['rows']]}")
        elif got["n_pop"] != 1:
            fails.append(f"population wrong: {got['n_pop']}")

        # nounset-only: PRECAUTIONARY, never EXPOSED (the T10 severity split —
        # measured, errexit is the precondition).
        (repo / "scripts" / "g.sh").write_text(
            exposed.replace("set -euo pipefail", "set -uo pipefail"))
        got = audit(repo)
        if len(got["precautionary"]) != 1 or got["exposed"]:
            fails.append(f"nounset-only must be PRECAUTIONARY, got "
                         f"{[r['verdict'] for r in got['rows']]}")

        # CONTROL: a sentinelled script must produce NO finding, or the sweep
        # reds on every correct repair and gets switched off.
        (repo / "scripts" / "g.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'A="$(mktemp)"\nDONE_FLAG=0\n'
            "cleanup() {\n    st=$?\n    rm -f \"$A\"\n"
            '    if [ "$DONE_FLAG" != "1" ] && [ "$st" = "0" ]; then\n'
            "        exit 1\n    fi\n    exit \"$st\"\n}\n"
            "trap cleanup EXIT\necho layer\nDONE_FLAG=1\n")
        got = audit(repo)
        if any(got[name] for name, _ in CLASSES):
            fails.append("control: a sentinelled script produced a finding "
                         f"({[r['verdict'] for r in got['rows']]})")
        if got["n_pop"] != 1:
            fails.append("control: the sentinelled script left the population")

        # population derivation: BOUNDARY.md + a .git DIRECTORY, both required
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if tela.repos(str(ws)) != ["fake-repo"]:
            fails.append(f"population derivation wrong: {tela.repos(str(ws))}")

        # pin parser: arity ENFORCED, comments stripped
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a 10 5 0 0 0 0 0  # trailing\n\n")
        want = {"repo-a": dict(zip(FIELDS, (10, 5, 0, 0, 0, 0, 0)))}
        if parse_pins(pins) != want:
            fails.append("pin parse: 8-field row not read correctly")
        pins.write_text("repo-a 1 2 3\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass
    return fails


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior

    # The classifier's own selftest FIRST. Without it a parser regression takes
    # every count to zero and this sweep certifies the workspace clean on the
    # strength of an instrument that has gone blind.
    blind = tela.selftest()
    if blind:
        print("✗ trap sweep SELFTEST FAILED — the classifier does not pass its own:")
        for f in blind:
            print(f)
        return 2

    fails = selftest()
    if fails:
        print("✗ trap sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    # The premise is measured, not quoted — and if THIS box launders nothing,
    # every EXPOSED row below is a finding about a premise that does not hold
    # here. Say so rather than printing the rows as though it did.
    launders = [r for r in tela.measure_premise() if r["launders"]]
    diverged = [r for r in tela.measure_premise() if not r["as_documented"]]

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        pins = parse_pins(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    names = tela.repos(str(WORKSPACE))
    if not names:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    # Same quantity, two files: this sweep re-states katgpt-rs's population
    # floor that the per-push gate owns. Asserted, not trusted — the pattern
    # docs_gate_paths_sync.py uses for the two trigger lists.
    mine = pins.get(REPO_ROOT.name, {}).get("min_population")
    if mine != tsg.POPULATION_FLOOR:
        print(f"✗ pin drift: {PINS.name} says min_population={mine} for "
              f"{REPO_ROOT.name}, trap_sentinel_gate.POPULATION_FLOOR is "
              f"{tsg.POPULATION_FLOOR}. Same quantity, two files — change both.")
        return 1

    bad = False
    tot = {"n_scripts": 0, "n_pop": 0, "replaced": 0,
           "launderable": 0, "errexit": 0}
    for name, _ in CLASSES:
        tot[name] = 0

    for name in names:
        got = audit(WORKSPACE / name)
        row = pins.get(name)
        tot["n_scripts"] += got["n_scripts"]
        tot["n_pop"] += got["n_pop"]
        tot["replaced"] += got["replaced"]
        tot["errexit"] += sum(1 for r in got["rows"] if r["errexit"])
        # The bottom line, per Issue 734 T10: unsentinelled AND with at least
        # one abort SITE in the window [last trap registration, EOF). A
        # zero-trigger window provably cannot launder.
        tot["launderable"] += sum(
            1 for r in got["rows"]
            if r["verdict"] in (tela.EXPOSED, tela.LIVE_FORWARD) and r["triggers"] > 0)
        for cls, _ in CLASSES:
            tot[cls] += len(got[cls])

        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_scripts"] < row["min_scripts"]:
                flags.append(f"walk FLOOR breached: {got['n_scripts']} tracked "
                             f"*.sh < {row['min_scripts']} — scripts were "
                             f"removed, or walk_sh/git ls-files went blind")
            if got["n_pop"] < row["min_population"]:
                flags.append(f"population FLOOR breached: {got['n_pop']} < "
                             f"{row['min_population']} — a script lost its "
                             f"`set -e`/`set -u` or its EXIT trap, or the "
                             f"classifier stopped seeing them")
            for cls, _ in CLASSES:
                if len(got[cls]) > row[f"max_{cls}"]:
                    flags.append(f"{cls} {len(got[cls])} > pinned {row[f'max_{cls}']}")
            if got["replaced"] > row["max_replaced"]:
                flags.append(f"replaced {got['replaced']} > pinned "
                             f"{row['max_replaced']}")

        findings = [r for cls, _ in CLASSES for r in got[cls]]
        status = "✗" if flags else ("·" if findings else "✓")
        print(f"{status} {name:22s} sh={got['n_scripts']:<4d} pop={got['n_pop']:<3d} "
              f"exposed={len(got['exposed'])} precautionary="
              f"{len(got['precautionary'])} live_forward="
              f"{len(got['live_forward'])} unparsed={len(got['unparsed'])} "
              f"replaced={got['replaced']}")
        for r in sorted(findings, key=lambda r: -r["triggers"]):
            inert = "  <- ZERO abort sites in the window: provably cannot launder" \
                if r["triggers"] == 0 else ""
            print(f"      {r['file']}  {r['verdict']}  "
                  f"[window {r['window']} line(s), {r['triggers']} trigger(s)]{inert}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - set(names)):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(names)} contract repo(s) · {tot['n_scripts']} tracked *.sh · "
          f"{tot['n_pop']} in population ({tot['errexit']} with errexit) · "
          f"{tot['exposed']} exposed · {tot['precautionary']} precautionary · "
          f"{tot['live_forward']} live-forward · {tot['unparsed']} unparsed · "
          f"{tot['replaced']} replaced")
    print(f"  bottom line: {tot['launderable']} script(s) are unsentinelled AND "
          f"have an abort site in the window — those are the only ones that can "
          f"report a PASS on an abort today.")
    # State the scope where it is READ, not only in the docstring.
    print("  scope: PRECAUTIONARY is nounset WITHOUT errexit — measured, those "
          "aborts exit 1, so it is not laundering today and is pinned "
          "separately rather than pooled into exposed.")
    if not launders:
        print("  ⛔ this bash LAUNDERS NOTHING in any measured arm — every row "
              "above is about a premise that does not hold on this box. "
              "Re-read before acting.")
    if diverged:
        print(f"  ⛔ {len(diverged)} premise cell(s) DIVERGE from the documented "
              f"table; run trap_exit_launder_audit.py for the matrix.")

    if bad:
        print("✗ trap sentinel sweep FAILED — see the ✗ rows above")
        print("    A gate that ABORTS mid-run reports exit 0. The repair is a "
              "completion sentinel; see scripts/full_gate.sh (full_gate_cleanup).")
        return 1
    print("✓ trap sentinel sweep PASSED — every repo at or under its pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
