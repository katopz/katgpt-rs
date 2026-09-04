#!/usr/bin/env python3
"""Which contract repos automatically EXECUTE a test, and which only compile one.

`scripts/ci_gate_coverage.py` answers "does anything automatically start this
repo's compile/lint surface". It deliberately does not ask whether anything
*runs* a test, and the two questions have different answers — measured
2026-09-03, katgpt-rs is fully covered by the first and scores ZERO on the
second: `cargo clippy` and `cargo check` compile every test target and execute
none, so 477 integration-test targets, 31 lib targets and 176 bench targets
are executed by nothing automatic (`.docs/10_audits/ci_compile_vs_execute_axis.md`,
the durable record of Issue 718, closed + removed 2026-09-04).

That is the fourth rung on a ladder this workspace has climbed three of:

  1. a workflow file is identical on disk whether or not it can execute
     (`.issues/704` — inert on a non-default branch)
  2. "can fire is not does fire" (`.issues/706` — a dispatch-only gate is a
     button, not a schedule)
  3. "a green test count can be a count of nothing" (`713` — a `#![cfg]`-gated
     file compiles empty and prints `ok. 0 passed`)
  4. **"compiles is not runs"** — this report. Worse than a green zero,
     because there is no count at all: nothing produced one.

A **report, not a gate** (always exit 0), for the same reason
`ci_gate_coverage.py` is: a repo can have a legitimate reason to compile
without running (no runnable tests; a hardware-bound suite), and a report that
exits 1 on those is a report nobody runs.

**Read the split, never a pooled total.** COMPILE-ONLY over a repo with 500
test targets and COMPILE-ONLY over a repo with none are the same word and
opposite facts, which is why `#[test]` sites are printed beside the verdict
(the `713` lesson: a pooled figure read 702 and meant nothing).

Vocabulary is DATA and committed (`EXEC`/`COMPILE_ONLY` below), population is
DERIVED (BOUNDARY.md + a `.git` dir), per the workspace rule that deriving both
from one walk is what makes a cross-repo report permanently empty.

    scripts/ci_test_execution_report.py             # all contract repos
    scripts/ci_test_execution_report.py ../riir-ai  # or one, by path
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_gate_coverage import (  # noqa: E402
    _tracked,
    live_lines,
    _wf_pool,
    cargo_commands,
    default_branch,
    derive_repos,
    reachable_triggers,
)

GIT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Vocabulary (data, exhaustive) ────────────────────────────────────────────
#
# EXEC: subcommands that RUN a compiled test binary.
# The `--no-run` / `--list` suppressors are the trap this vocabulary exists to
# survive: `cargo test --no-run` matches "cargo test" and executes NOTHING, so
# a token-only matcher credits a repo for a compile. Pinned by selftest().
EXEC = (
    "cargo test",
    "cargo nextest run",
    "cargo miri test",
    "cargo bench",
    "cargo llvm-cov",
    "cargo insta test",
)
SUPPRESS = ("--no-run", "--list", "--help")
COMPILE_ONLY = ("cargo check", "cargo clippy", "cargo build", "cargo doc", "cargo fmt")

TEST_ATTR = re.compile(r"#\[\s*(?:tokio::)?test\b|#\[\s*bench\b")
SKIP_DIRS = {"target", ".git", "node_modules", "vendor", ".venv"}

_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_SUBST = re.compile(r"\$\(([^()]*)\)")


def invocation_texts(cmd: str) -> list[str]:
    """The spans of `cmd` where a `cargo` mention is a real INVOCATION.

    Necessary because half this workspace's guard scripts announce their layers
    with the command name: `echo "── L4: cargo test (default features) ──"`,
    `layer "4/4 cargo test --workspace ..."`, `fail "cargo test --workspace"`.
    A token-only matcher credits every one of those as an executed test, and
    a repo whose ONLY match is a label would be reported EXECUTES while running
    nothing — the exact false green this report exists to find.

    A label's `cargo` is always inside a quoted string; a real invocation's
    never is. So: lift `$(...)` contents out first (they ARE command context
    even when the substitution sits inside quotes — riir-chain's
    `out="$(cargo test ...)"` is a real run), then strip quoted literals from
    what remains."""
    subs = [m.group(1) for m in _SUBST.finditer(cmd)]
    outer = _QUOTED.sub(" ", _SUBST.sub(" ", cmd))
    return [outer, *subs]


def classify(cmd: str) -> str:
    """`exec` | `compile` | `other` for one cargo invocation."""
    verdict = "other"
    for text in invocation_texts(cmd):
        if any(tok in text for tok in EXEC):
            # A suppressed run compiles and exits — a COMPILE, not a run.
            if not any(sup in text for sup in SUPPRESS):
                return "exec"
            verdict = "compile"
        elif any(tok in text for tok in COMPILE_ONLY):
            verdict = "compile"
    return verdict


def live_and_automatic(trig: set[str]) -> tuple[set[str], bool]:
    """Split `reachable_triggers`' return into (live, is-automatic).

    `reachable_triggers` MIXES live triggers with negative `-<trigger>` markers
    for declared-but-dead ones, so a workflow whose every trigger is dead comes
    back NON-EMPTY. Read naively, `riir-game-sdk/nightly.yml` — which
    `ci_gate_coverage.py` correctly calls UNREACHABLE — reported as executing
    tests. Two instruments disagreed and the newer one was wrong.

    `pull_request?` is deliberately NOT automatic: whether a PR is ever opened
    is workflow policy git cannot see, and several repos here land directly on
    develop. `workflow_dispatch` alone is a button, not automation (706)."""
    live = {t for t in trig if not t.startswith("-")}
    return live, bool(live - {"workflow_dispatch", "pull_request?"})


def test_sites(repo_dir: Path) -> int:
    """`#[test]` / `#[bench]` attribute count — the assertions at risk.

    A count, not a list: its only job is to separate "compiles a large suite it
    never runs" from "has nothing to run", which are the same verdict word and
    opposite facts."""
    n = 0
    for p in repo_dir.rglob("*.rs"):
        if SKIP_DIRS & set(p.parts):
            continue
        try:
            n += len(TEST_ATTR.findall(p.read_text(errors="replace")))
        except OSError:
            continue
    return n


def survey_repo(root: Path, repo: str) -> dict:
    wf_dir = root / repo / ".github" / "workflows"
    dflt = default_branch(root, repo)
    live_exec: list[tuple[str, str]] = []   # (workflow, command)
    live_compile: list[str] = []
    dispatch_only_exec: list[tuple[str, str]] = []
    workflows = sorted(wf_dir.glob("*.yml")) if wf_dir.is_dir() else []

    for wf in workflows:
        rel = str(wf.relative_to(root / repo))
        if not _tracked(root, repo, rel):
            continue  # a colleague's in-flight file is not a defect
        live, automatic = live_and_automatic(
            reachable_triggers(root, repo, wf, dflt))
        if not live:
            continue
        for cmd in cargo_commands(_wf_pool(root, repo, wf)):
            kind = classify(cmd)
            if kind == "exec":
                (live_exec if automatic else dispatch_only_exec).append((wf.name, cmd))
            elif kind == "compile" and automatic:
                live_compile.append(cmd)

    sites = test_sites(root / repo)
    if dflt == "?":
        verdict = "UNMEASURED"
    elif live_exec:
        verdict = "EXECUTES"
    elif dispatch_only_exec:
        verdict = "BUTTON-ONLY"
    elif sites == 0:
        verdict = "NOTHING-TO-RUN"
    elif live_compile:
        verdict = "COMPILE-ONLY"
    else:
        verdict = "NO-CARGO"
    return {
        "repo": repo, "verdict": verdict, "sites": sites,
        "exec": live_exec, "dispatch_only": dispatch_only_exec,
        "compiles": len(live_compile), "workflows": len(workflows),
    }


def selftest() -> None:
    """Pins the classifier. Exits 2 rather than printing: a degraded matcher
    reports every repo as NOTHING-TO-RUN or NO-CARGO, which is a confident
    green indistinguishable from a real one."""
    cases = [
        # The trap: matches "cargo test" and runs nothing.
        ("cargo test --workspace --no-run", "compile"),
        ("cargo test --no-run --all-features", "compile"),
        ("cargo test --list", "compile"),
        # Real executions.
        ("cargo test --workspace --all-features", "exec"),
        ("cargo test -p katgpt-core --lib", "exec"),
        ("cargo nextest run --workspace", "exec"),
        ("cargo bench --bench foo", "exec"),
        ("cargo miri test", "exec"),
        # LABELS, not invocations — the guard scripts announce their layers.
        ('echo "── L4: cargo test (default features) ──"', "other"),
        ('layer "4/4 cargo test --workspace (incl. the local e2e)"', "other"),
        ('ok "cargo test --workspace"', "other"),
        # A real run whose FAILURE message names itself.
        ('cargo test --workspace --quiet || fail "cargo test --workspace"', "exec"),
        # A real run inside a command substitution inside quotes (riir-chain).
        ('if ! out="$(cargo test --features "$f" --test "$n" 2>&1)"; then', "exec"),
        # Compile-only surfaces.
        ("cargo clippy --workspace --all-targets --all-features", "compile"),
        ("cargo check --release", "compile"),
        ("cargo build --locked", "compile"),
        ("cargo fmt --check", "compile"),
        # Neither.
        ("cargo metadata --no-deps", "other"),
        ("cargo install cargo-nextest", "other"),
    ]
    for cmd, want in cases:
        got = classify(cmd)
        if got != want:
            print(f"SELFTEST FAIL: {cmd!r} -> {got}, want {want}", file=sys.stderr)
            raise SystemExit(2)
    # A comment mentioning a test run must not count. The stripping happens in
    # `live_lines`, NOT in `cargo_commands` — a first cut pinned the wrong
    # function and this selftest failed on its first run, which is the point.
    # Pin the COMPOSITION, since that is what the report is downstream of.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        fh.write("# run cargo test --workspace here\n"
                 "  # cargo nextest run --workspace\n"
                 "        run: cargo clippy --workspace\n")
        tmp = Path(fh.name)
    try:
        cmds = cargo_commands(live_lines(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    if [classify(c) for c in cmds] != ["compile"]:
        print(f"SELFTEST FAIL: commented cargo test lines were counted -> {cmds}",
              file=sys.stderr)
        raise SystemExit(2)
    # And the join must survive a continuation, or a multi-line run: block
    # silently classifies as `other`.
    joined = cargo_commands(["cargo test --workspace \\", "  --all-features"])
    if len(joined) != 1 or classify(joined[0]) != "exec":
        print(f"SELFTEST FAIL: continuation join -> {joined}", file=sys.stderr)
        raise SystemExit(2)
    # The negative-marker leak. A workflow whose every declared trigger is
    # DEAD comes back non-empty from reachable_triggers.
    for trig, want in (
        ({"-schedule", "-workflow_dispatch"}, (set(), False)),
        ({"workflow_dispatch", "-push"}, ({"workflow_dispatch"}, False)),
        ({"pull_request?"}, ({"pull_request?"}, False)),
        ({"schedule", "-push"}, ({"schedule"}, True)),
        ({"push"}, ({"push"}, True)),
    ):
        if live_and_automatic(trig) != want:
            print(f"SELFTEST FAIL: live_and_automatic({trig}) -> "
                  f"{live_and_automatic(trig)}, want {want}", file=sys.stderr)
            raise SystemExit(2)


def main(argv: list[str]) -> int:
    selftest()
    if len(argv) > 1:
        target = Path(argv[1]).resolve()
        root, repos = target.parent, [target.name]
    else:
        root, repos = GIT_ROOT, derive_repos(GIT_ROOT)

    rows = [survey_repo(root, r) for r in repos]
    print(f"\nci test-execution report — {len(rows)} repo(s) "
          f"(derived: BOUNDARY.md + .git)\n")
    print(f"  {'repo':<24}{'verdict':<17}{'#[test] sites':>14}{'wf':>5}"
          f"{'auto compiles':>15}")
    for r in sorted(rows, key=lambda r: (r["verdict"], r["repo"])):
        print(f"  {r['repo']:<24}{r['verdict']:<17}{r['sites']:>14}"
              f"{r['workflows']:>5}{r['compiles']:>15}")

    worst = [r for r in rows if r["verdict"] == "COMPILE-ONLY"]
    if worst:
        print("\n── COMPILE-ONLY: every test target is compiled and none is run ──")
        for r in sorted(worst, key=lambda r: -r["sites"]):
            print(f"  {r['repo']}: {r['sites']} #[test] site(s) executed by nothing "
                  f"automatic ({r['compiles']} automatic compile command(s))")
    button = [r for r in rows if r["verdict"] == "BUTTON-ONLY"]
    if button:
        print("\n── BUTTON-ONLY: a test run exists but only `workflow_dispatch` "
              "starts it (Issue 706) ──")
        for r in button:
            for wf, cmd in r["dispatch_only"][:3]:
                print(f"  {r['repo']}/{wf}: {cmd[:88]}")
    good = [r for r in rows if r["verdict"] == "EXECUTES"]
    if good:
        print("\n── EXECUTES: something automatic runs a test ──")
        for r in good:
            wf, cmd = r["exec"][0]
            print(f"  {r['repo']}/{wf}: {cmd[:88]}"
                  f"{'  (+%d more)' % (len(r['exec']) - 1) if len(r['exec']) > 1 else ''}")

    print("\n  NOTHING-TO-RUN = 0 #[test] sites, so compile-only is correct there.")
    print("  Report only; exit 0 always. `--no-run` counts as a COMPILE, not a run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
