#!/usr/bin/env python3
"""rustfmt-clean, with a pinned drain list — Issue 946 (owner verdict: option 1).

The repo is rustfmt-DEFAULT-styled (no `rustfmt.toml` tuning: the Issue 946
residual curve measured every wider setting unpacking 2,000+ default-styled
files). 158 of 2,770 tracked `.rs` had drifted — files edited without ever
running fmt. 131 were formatted in one commit (listed in
`.git-blame-ignore-revs`); the 26 touched by a commit in the 48 h before that
landing were HELD BACK, because reformatting a file a concurrent session is
mid-edit on turns its next rebase into a conflict. Those 26 are the DRAIN LIST
in `scripts/fmt_pending_expected.txt`.

Verdict, by MEMBERSHIP, both directions (a count is green on a swap):

- **UNPINNED** — a file `cargo fmt --check` reports that no row names: a NEW
  drift. Run `cargo fmt --all` (or `rustfmt --edition 2024 <file>` — never
  `cargo fmt -p X -- file`, which formats the whole package) and commit.
- **STALE** — a row whose file is now clean (or gone): delete the row in the
  commit that drained it, so the list can only shrink.

Two floors, because a silent formatter reads exactly like a clean tree: the
run must REPORT formatting at least `MIN_FILES` distinct files (`-v`), and a
non-0/1 exit is NO VERDICT (exit 2), never a pass.

Runs from `scripts/full_gate.sh` Layer 7, not the docs gate: docs_gate.yml is
deliberately toolchain-free. ~13 s on the M3.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_safe  # noqa: E402

console_safe.apply()

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "scripts" / "fmt_pending_expected.txt"
# Measured 2026-10-17: 2,770 tracked .rs, ~2,9xx distinct files formatted.
MIN_FILES = 2500


def load_pins(path: Path) -> tuple[set[str], list[str]]:
    """Rows are `<relpath> # reason`. A reasonless row is refused."""
    pins: set[str] = set()
    errors: list[str] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rel, sep, reason = line.partition("#")
        if not sep or not reason.strip():
            errors.append(f"{path.name}:{n}: row has no reason: {line!r}")
            continue
        pins.add(rel.strip())
    return pins, errors


def normalise(root: Path, abs_path: str) -> str:
    # rustfmt reports a `#[path]`/`mod` reached through `..` verbatim
    # (`benches/../tests/common/x.rs`); one file, one key.
    return Path(os.path.relpath(os.path.normpath(abs_path), root)).as_posix()


def parse_fmt_output(root: Path, out: str) -> tuple[set[str], set[str]]:
    """(diverging files, files formatted at all) from `--check -l -v` stdout."""
    diverging: set[str] = set()
    seen: set[str] = set()
    for line in out.splitlines():
        if line.startswith("Formatting "):
            seen.add(normalise(root, line[len("Formatting ") :].strip()))
        elif line.startswith("/") and line.endswith(".rs"):
            diverging.add(normalise(root, line.strip()))
    return diverging, seen


def verdict(diverging: set[str], pins: set[str]) -> tuple[list[str], list[str]]:
    """(unpinned, stale) — the whole decision, extracted so the arms reach it."""
    return sorted(diverging - pins), sorted(pins - diverging)


def selftest() -> list[str]:
    fails: list[str] = []
    root = Path("/r")
    out = (
        "Formatting /r/a.rs\nFormatting /r/b/../c.rs\n/r/a.rs\n"
        "/r/b/../c.rs\nSpent 0.001 secs in the parsing phase\n"
    )
    div, seen = parse_fmt_output(root, out)
    if div != {"a.rs", "c.rs"} or seen != {"a.rs", "c.rs"}:
        fails.append(f"parse: {div=} {seen=}")
    if verdict({"a.rs"}, {"a.rs"}) != ([], []):
        fails.append("exact match must be clean")
    if verdict({"a.rs", "n.rs"}, {"a.rs"}) != (["n.rs"], []):
        fails.append("a new drift must be UNPINNED")
    if verdict(set(), {"a.rs"}) != ([], ["a.rs"]):
        fails.append("a drained row must be STALE")
    if verdict({"x.rs"}, {"y.rs"}) != (["x.rs"], ["y.rs"]):
        fails.append("a swap must red both ways (membership, not count)")
    return fails


def main() -> int:
    fails = selftest()
    if fails:
        for f in fails:
            print(f"✗ fmt gate self-test: {f}")
        return 2

    pins, errors = load_pins(PINS)
    if errors:
        for e in errors:
            print(f"✗ {e}")
        return 1

    proc = subprocess.run(
        ["cargo", "fmt", "--all", "--", "--check", "-l", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode not in (0, 1):
        print(f"✗ fmt gate NO VERDICT — cargo fmt exited {proc.returncode}")
        print(proc.stderr.strip()[-2000:])
        return 2
    diverging, seen = parse_fmt_output(ROOT, proc.stdout)
    if len(seen) < MIN_FILES:
        print(f"✗ fmt gate BLIND — rustfmt reported {len(seen)} file(s) < floor {MIN_FILES}")
        return 2
    if proc.returncode == 1 and not diverging:
        print("✗ fmt gate NO VERDICT — cargo fmt exited 1 but listed no file")
        return 2

    unpinned, stale = verdict(diverging, pins)
    for f in unpinned:
        print(f"  ✗ UNPINNED {f} — run `rustfmt --edition 2024 {f}` and commit it")
    for f in stale:
        print(f"  ✗ STALE {f} — now rustfmt-clean (or gone); delete its row in {PINS.name}")
    if unpinned or stale:
        print(f"✗ fmt gate FAILED — {len(unpinned)} unpinned, {len(stale)} stale over {len(seen)} files")
        return 1
    print(f"✓ fmt gate PASSED — {len(seen)} files checked, {len(pins)} pinned on the drain list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
