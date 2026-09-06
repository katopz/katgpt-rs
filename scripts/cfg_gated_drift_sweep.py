#!/usr/bin/env python3
"""Run the cfg-gated silent-zero ceilings over EVERY contract repo, not just this one.

`scripts/cfg_gated_floor_gate.py` is katgpt-rs-scoped by construction: its pins
live in `scripts/cfg_gated_floors.txt`, and `docs_gate.yml` has a single
checkout so it could never see a sibling. That is the right shape for a
per-push CI gate and the wrong shape for "is anyone ELSE shipping a
`#![cfg]`-gated gate that prints `ok. 0 passed` over an empty binary?"

That question had never been asked with a verdict attached, and asking it is
what found Issue 728: run over all 16 repos, the load-bearing classifier
reported **0 in every one**, because its token set had been validated against
2,157 katgpt-rs-era target names while the workspace corpus is 3,081. Six
tokens and one compound later the count is **12**, two of them in katgpt-rs
itself. This sweep is what keeps it from becoming 13.

The fourth and last member of the sweep family:

    Issue 702  ci_gate_coverage              one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep         one repo -> 35 duplicate numbers
    2026-09-06 required_features_drift_sweep one repo -> clean, and pinned there
    2026-09-06 percentile_drift_sweep        one repo -> clean, and pinned there
    this file  cfg_gated_drift_sweep         one repo -> 12 instances, ratcheted

Ceilings here are a RATCHET, not a wall — the difference from the two sweeps
that landed the same day
--------------------------------------------------------------------------
Those two measured zero everywhere, so their ceilings are walls. This one has a
standing backlog of 12 across six repos, ten of which are sibling-owned and are
NOT this session's to arm. So each repo's ceiling is pinned at its MEASURED
count, exactly as `numbering_drift_floors.txt` does: a new instance reds
immediately, and the backlog is visible in the pins rather than silently
tolerated. Lower a pin in the commit that arms a target.

Read the SEVERITY SPLIT, never the pooled total
-----------------------------------------------
`silent_now` is the severe class — those targets zero on a **plain `cargo
test`**. `latent` ones vanish only under `--no-default-features` and are not
gated here. And `silent_now_load_bearing` is the column that decides whether a
green is EVIDENCE: a silent zero on `scratch_probe` costs a reader's time; a
silent zero on `bridge_spec_match` is a promotion argument resting on an empty
binary. Both are pinned, because they fail independently.

Why this is NOT in scripts/docs_gate.sh's CHECKS
-----------------------------------------------
Identical to the other three sweeps: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script                 workstation, on demand, every contract repo
    cfg_gated_floor_gate.py     CI, per-push (docs_gate.sh), katgpt-rs only

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy** — an unreliable instrument is not the same finding as drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the scanner, the classifier and the selftest are the report's, so the
# sweep and the per-push gate can never disagree about what is load-bearing.
import cfg_gated_target_audit as cga  # noqa: E402

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "cfg_gated_drift_floors.txt"
# The per-push gate's pins. This sweep re-states katgpt-rs's four numbers, so
# the two files can drift apart — asserted below rather than trusted, exactly
# as docs_gate_paths_sync.py does for the two trigger lists.
LOCAL_PINS = HERE / "cfg_gated_floors.txt"

FIELDS = ("min_targets", "min_gated", "max_silent_now", "max_load_bearing")
# Every field here names the SAME quantity as the identically-named key in
# cfg_gated_floors.txt, which is what makes the sync assert total rather than
# a spot check on one number.
SYNCED = FIELDS


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


def local_pins(path: Path) -> dict[str, int]:
    """The per-push gate's pins — `key<TAB>value`, comments stripped."""
    out: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def audit(repo: Path) -> dict:
    rep = cga.audit(repo)
    sn = rep.silent_now()
    return {
        "n_targets": rep.scanned,
        "n_gated": rep.gated,
        "silent_now": sn,
        "load_bearing": [f for f in sn if cga.is_load_bearing(f.name)],
    }


def selftest() -> list[str]:
    """Pin that the verdict FIRES, that the severity split survives, and the
    parsers. A silent failure here reports a clean workspace."""
    import tempfile

    fails = []
    # The classifier's own widening (Issue 728) is what this sweep gates on.
    # If a future edit narrows it back, every ceiling below goes green over a
    # smaller population — the exact failure the issue documents. Pin the two
    # katgpt-rs instances by NAME, and the homonyms that must stay excluded.
    for name in ("bridge_spec_match", "pencil_spec_match",
                 "gemma4_q4k_gguf_parity", "kat_grant_reachability",
                 "net_ffi_roundtrip", "ledger_exactness"):
        if not cga.is_load_bearing(name):
            fails.append(f"classifier narrowed: {name!r} no longer load-bearing")
    for name in ("spec_reconciliation_bench", "spec_reconciliation_demo",
                 "attn_match_online", "quest_match_tui", "checkpoint_cost",
                 "aggregate_delegate_propagate", "bench_578_mcts_budget_sweep"):
        if cga.is_load_bearing(name):
            fails.append(f"classifier over-wide: {name!r} wrongly load-bearing")

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = ws / "fake-repo"
        (repo / "tests").mkdir(parents=True)
        (repo / "BOUNDARY.md").write_text("x")
        (repo / ".git").mkdir()
        (repo / "Cargo.toml").write_text(
            "[package]\nname = 'fake'\nversion = '0.1.0'\n[features]\noff = []\n"
        )
        # Auto-discovered, gated on a DEFAULT-OFF feature, load-bearing name.
        (repo / "tests" / "thing_spec_match.rs").write_text(
            '#![cfg(feature = "off")]\n#[test]\nfn t() { assert!(true); }\n'
        )
        # Same shape, NOT load-bearing by name — must count as silent_now but
        # not as load_bearing, or the severity split has collapsed.
        (repo / "tests" / "scratch_probe.rs").write_text(
            '#![cfg(feature = "off")]\n#[test]\nfn t() { assert!(true); }\n'
        )
        got = audit(repo)
        if len(got["silent_now"]) != 2:
            fails.append(f"expected 2 SILENT-NOW, got "
                         f"{[f.name for f in got['silent_now']]}")
        if [f.name for f in got["load_bearing"]] != ["thing_spec_match"]:
            fails.append(f"severity split broken: "
                         f"{[f.name for f in got['load_bearing']]}")

        # population derivation: BOUNDARY.md + a .git DIRECTORY, both required
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if [p.name for p in cga.derive_repos(ws)] != ["fake-repo"]:
            fails.append("population derivation is not BOUNDARY.md + .git dir")

        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a 10 5 3 0  # trailing\n\n")
        if parse_pins(pins) != {"repo-a": {"min_targets": 10, "min_gated": 5,
                                           "max_silent_now": 3,
                                           "max_load_bearing": 0}}:
            fails.append("pin parse: 5-field row not read correctly")
        pins.write_text("repo-a 1 2\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass
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
    # The report's own selftest raises a bare AssertionError rather than
    # exiting, so let it through and this sweep dies with a traceback — which
    # reads as a crash, not as the "instrument untrustworthy" verdict it IS.
    # Exit 2 is the whole point of having a third exit code.
    try:
        cga.selftest()
    except AssertionError as e:
        print("✗ cfg-gated sweep SELFTEST FAILED — the REPORT's own selftest "
              "does not hold, so no verdict is possible:")
        print(f"    {e}")
        return 2

    fails = selftest()
    if fails:
        print("✗ cfg-gated sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

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

    repos = cga.derive_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    # All four numbers are stated twice. A hand-duplicated pin drifts.
    lp = local_pins(LOCAL_PINS)
    mine = pins.get(REPO_ROOT.name)
    if mine is None:
        print(f"✗ {PINS.name} has no row for {REPO_ROOT.name}")
        return 2
    for key in SYNCED:
        if lp.get(key) != mine[key]:
            print(f"✗ pin drift: {PINS.name} says {key}={mine[key]} for "
                  f"{REPO_ROOT.name}, {LOCAL_PINS.name} says {key}="
                  f"{lp.get(key)}. Same quantity, two files — change both.")
            return 1

    seen = {p.name for p in repos}
    bad = False
    tot = {"n_targets": 0, "n_gated": 0, "silent_now": 0, "load_bearing": 0}

    for repo in repos:
        got = audit(repo)
        row = pins.get(repo.name)
        tot["n_targets"] += got["n_targets"]
        tot["n_gated"] += got["n_gated"]
        tot["silent_now"] += len(got["silent_now"])
        tot["load_bearing"] += len(got["load_bearing"])
        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_targets"] < row["min_targets"]:
                flags.append(f"target FLOOR breached: {got['n_targets']} < "
                             f"{row['min_targets']} — targets were removed, or "
                             f"the manifest walk went blind")
            if got["n_gated"] < row["min_gated"]:
                flags.append(f"gated FLOOR breached: {got['n_gated']} < "
                             f"{row['min_gated']} — or the #![cfg] scanner "
                             f"stopped recognising the shape")
            if len(got["silent_now"]) > row["max_silent_now"]:
                flags.append(f"SILENT-NOW {len(got['silent_now'])} > pinned "
                             f"{row['max_silent_now']}")
            if len(got["load_bearing"]) > row["max_load_bearing"]:
                flags.append(f"load-bearing SILENT-NOW {len(got['load_bearing'])} "
                             f"> pinned {row['max_load_bearing']} — a target whose "
                             f"NAME says its green is evidence reports "
                             f"`ok. 0 passed` over an EMPTY binary")
        status = "✗" if flags else ("·" if got["load_bearing"] else "✓")
        print(f"{status} {repo.name:22s} targets={got['n_targets']:<5d} "
              f"gated={got['n_gated']:<5d} silent_now={len(got['silent_now']):<3d} "
              f"load_bearing={len(got['load_bearing'])}")
        for f in got["load_bearing"]:
            print(f"      load-bearing: {f.kind}:{f.name}  features={f.features}  "
                  f"({f.reason})")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - seen):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(repos)} contract repo(s) · {tot['n_targets']} target(s) · "
          f"{tot['n_gated']} #![cfg]-gated · {tot['silent_now']} SILENT-NOW · "
          f"{tot['load_bearing']} of those load-bearing")
    # State the scope where it is READ. A pooled count means nothing here.
    print("  scope: SILENT-NOW only — targets that zero on a PLAIN `cargo "
          "test`. Targets gated on a DEFAULT-ON feature (`latent`) vanish only "
          "under --no-default-features and are not gated by this sweep; nor "
          "are target_os/miri/any(...) gates, which required-features cannot "
          "express. A ceiling here is a RATCHET at the measured backlog, not a "
          "claim of zero.")
    if bad:
        print("✗ cfg-gated sweep FAILED — see the ✗ rows above")
        print("    The fix is a `required-features` row: the #![cfg] protects")
        print("    the COUNT, required-features protects the READER. Adding one")
        print("    cannot red an existing CI — cargo SKIPS a target whose")
        print("    features are unmet; it only stops the green zero.")
        return 1
    print("✓ cfg-gated sweep PASSED — nothing above its pinned ratchet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
