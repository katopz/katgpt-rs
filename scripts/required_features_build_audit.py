#!/usr/bin/env python3
"""Ask the COMPILER whether each `required-features` row can actually build.

`scripts/cfg_gated_target_audit.py` answers "does this target have a
`required-features` row?" — the right question for the failure it was built
for (a `#![cfg]`-gated target with no row compiles to an empty binary and
prints a green `ok. 0 passed`). It says nothing about whether the row is
CORRECT, and a row that exists and is wrong is strictly worse than a missing
one:

- `cargo test --workspace` silently SKIPS the target (features unmet), so
  nothing reds.
- `--all-features` builds it, because the union supplies whatever the row
  forgot — so the one configuration anybody runs it in passes.
- Every audit counts it in the "w/ req-f" column, i.e. **protected**.

Measured instance (riir-train `9da3420f`): `test_cubecl_backward_grads`
declared `["cubecl_runtime", "gemma_lora", "moved-gpu-tests"]` and omitted
`gpu_training_resident`, which its own import needs. Naming the target with
its own row is an `E0432`. Fixing the row made it build and immediately
reported 9 passed / 1 failed — the wrong row was hiding a real defect, not
merely a build error.

## Why this cannot be a static check

The row is wrong relative to what the file *imports*, and the import resolves
through `lib.rs` re-exports that are themselves cfg-gated. Chasing that
statically is the glob-re-export problem the workspace already documents as
defeating grep. The only sound check is to run

    cargo check -p <pkg> --test <name> --features <the row, verbatim>

per target — which is why this is an on-demand **report** (always exit 0),
not a per-push gate. It is expensive: one build per target.

## What a green run does NOT say

Nothing here executes anything, and `cargo check` does not link. A BUILDS
verdict means the row is *sufficient to typecheck the target*, not that its
assertions pass or that the feature set is the one the author intended. A row
can also be over-broad — enabling features the target does not need — and
this report cannot see that; it is the under-specified direction that
silently reports green.

## Population is derived, expectations are committed

The repo set comes from the workspace walk (a root `BOUNDARY.md` **and** a
`.git` dir), never a typed list — deriving both the population and the
expectation from the same walk is what makes a cross-repo report permanently
empty.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# DRY: the population walk and the manifest list are the sibling auditor's,
# so the two reports can never disagree about WHICH repos or manifests exist.
from cfg_gated_target_audit import derive_repos, manifests  # noqa: E402

KIND_FLAG = {"test": "--test", "bench": "--bench", "example": "--example"}
KIND_DIR = {"test": "tests", "bench": "benches", "example": "examples"}

# Verdicts, most severe first. BUILDS is the only clean one.
FAILS = "FAILS-TO-BUILD"
NO_FEATURE = "NO-SUCH-FEATURE"
BUILDS = "BUILDS"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"


@dataclass
class Row:
    repo: str
    package: str
    kind: str
    name: str
    features: list[str]
    path: str

    @property
    def label(self) -> str:
        return f"{self.repo}:{self.package}:{self.kind}:{self.name}"


@dataclass
class Result:
    row: Row
    verdict: str
    seconds: float
    detail: str = ""


@dataclass
class RepoReport:
    repo: str
    rows: list[Row] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(1 for r in self.results if r.verdict == verdict)


def parse_rows(repo: Path) -> list[Row]:
    """Every `[[test]]`/`[[bench]]`/`[[example]]` row carrying required-features.

    Rows WITHOUT the key are out of scope by construction: this report audits
    whether a declared row builds, not whether a row should exist. That second
    question is `cfg_gated_target_audit.py`'s, and keeping them apart is what
    lets each one be read as a claim about one thing.
    """
    out: list[Row] = []
    for manifest in manifests(repo):
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8", errors="replace"))
        except (tomllib.TOMLDecodeError, OSError):
            continue
        pkg = (data.get("package") or {}).get("name")
        if not pkg:
            continue  # virtual workspace root — its members are their own manifests
        for kind in KIND_FLAG:
            for entry in data.get(kind, []) or []:
                if not isinstance(entry, dict):
                    continue
                feats = entry.get("required-features")
                name = entry.get("name")
                if not feats or not name:
                    continue
                rel = entry.get("path") or f"{KIND_DIR[kind]}/{name}.rs"
                out.append(
                    Row(
                        repo=repo.name,
                        package=pkg,
                        kind=kind,
                        name=name,
                        features=list(feats),
                        path=str((manifest.parent / rel).resolve()),
                    )
                )
    return out


def classify(proc: subprocess.CompletedProcess[str]) -> tuple[str, str]:
    err = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode == 0:
        return BUILDS, ""
    # A feature named in the row that the package does not define. Distinct
    # from a compile error: the row is not merely insufficient, it is invalid.
    for marker in (
        "none of the selected packages contains these features",
        "does not have the feature",
        "did not match any packages",
        "feature `",
    ):
        if marker in err and "error: " in err:
            first = next(
                (ln for ln in err.splitlines() if ln.startswith("error")), ""
            )
            if "none of the selected packages" in err or "does not have" in err:
                return NO_FEATURE, first.strip()
    codes = sorted(
        {
            ln.split("[", 1)[1].split("]", 1)[0]
            for ln in err.splitlines()
            if ln.startswith("error[") and "]" in ln
        }
    )
    first = next((ln for ln in err.splitlines() if ln.startswith("error")), "")
    detail = ",".join(codes) if codes else first.strip()[:160]
    return FAILS, detail


def check_row(repo: Path, row: Row, target_dir: str | None, timeout: int) -> Result:
    cmd = [
        "cargo",
        "check",
        "-p",
        row.package,
        KIND_FLAG[row.kind],
        row.name,
        "--features",
        ",".join(row.features),
    ]
    env = dict(os.environ)
    # Never colourise: every anchored counter in this family has been defeated
    # by ANSI escapes at least once (katgpt-rs `.issues/705`).
    env["CARGO_TERM_COLOR"] = "never"
    if target_dir:
        env["CARGO_TARGET_DIR"] = target_dir
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Result(row, TIMEOUT, time.monotonic() - t0, f"> {timeout}s")
    except OSError as e:
        return Result(row, ERROR, time.monotonic() - t0, str(e))
    verdict, detail = classify(proc)
    return Result(row, verdict, time.monotonic() - t0, detail)


def concurrent_cargo(repo: Path) -> bool:
    """A cargo process with its CWD inside this repo invalidates the verdict.

    Detected by working directory rather than by command-line pattern: a
    pattern is blind to a plain `cargo build` and conversely matches harmless
    sibling-repo runs whose command lines mention our crates (katgpt-rs
    AGENTS.md, riir-game-sdk `.issues/023` T4).
    """
    if not shutil.which("lsof"):
        return False
    try:
        out = subprocess.run(
            ["lsof", "-a", "-c", "cargo", "-d", "cwd", "-Fn", "+D", str(repo / "target")],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def audit(repo: Path, args: argparse.Namespace) -> RepoReport:
    rep = RepoReport(repo=repo.name)
    rows = parse_rows(repo)
    if args.package:
        rows = [r for r in rows if r.package == args.package]
    if args.kind:
        kinds = set(args.kind.split(","))
        rows = [r for r in rows if r.kind in kinds]
    if args.grep:
        rows = [r for r in rows if args.grep in r.name]
    rep.rows = rows
    if args.limit:
        rows = rows[: args.limit]
    for i, row in enumerate(rows, 1):
        res = check_row(repo, row, args.target_dir, args.timeout)
        rep.results.append(res)
        if not args.quiet:
            mark = "  " if res.verdict == BUILDS else "!!"
            print(
                f"{mark} [{i}/{len(rows)}] {res.verdict:<15} {res.seconds:6.1f}s "
                f"{row.label}"
                + (f"  ({res.detail})" if res.detail else ""),
                flush=True,
            )
    return rep


# ---------------------------------------------------------------------------


def selftest() -> None:
    """Pin the manifest parse. Runs on EVERY invocation.

    Without it a parse regression is silent in the direction that reads as good
    news: the row list goes empty and the report prints a confident zero
    findings, indistinguishable from a workspace whose rows all build. Exits
    **2**, not 1 — an untrustworthy instrument is not the same finding as a
    real one.
    """
    import tempfile

    cases = [
        # (toml, expected (kind, name, features, path-suffix) tuples)
        (
            '[package]\nname = "p"\n\n[[test]]\nname = "t1"\n'
            'required-features = ["a", "b"]\n',
            [("test", "t1", ["a", "b"], "tests/t1.rs")],
        ),
        # An explicit `path` need not resemble the name — the sibling auditor
        # documents four katgpt-rs targets with exactly this shape.
        (
            '[package]\nname = "p"\n\n[[bench]]\nname = "b_goat"\n'
            'path = "benches/b.goat.rs"\nrequired-features = ["f"]\n',
            [("bench", "b_goat", ["f"], "benches/b.goat.rs")],
        ),
        # No required-features → out of scope, not a finding.
        ('[package]\nname = "p"\n\n[[test]]\nname = "plain"\n', []),
        # An empty list is not a row: cargo treats it as no constraint, and
        # reporting it would be a build per target for a guaranteed BUILDS.
        (
            '[package]\nname = "p"\n\n[[test]]\nname = "e"\nrequired-features = []\n',
            [],
        ),
        # A virtual workspace root has no [package]; its members carry theirs.
        ('[workspace]\nmembers = ["crates/*"]\n', []),
        (
            '[package]\nname = "p"\n\n[[example]]\nname = "ex"\n'
            'required-features = ["x"]\n',
            [("example", "ex", ["x"], "examples/ex.rs")],
        ),
    ]
    for toml_text, expected in cases:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "Cargo.toml").write_text(toml_text, encoding="utf-8")
            got = [
                (r.kind, r.name, r.features, Path(r.path).as_posix())
                for r in parse_rows(repo)
            ]
            want = [
                (k, n, f, (repo / s).resolve().as_posix()) for k, n, f, s in expected
            ]
            if got != want:
                print(
                    f"SELFTEST FAILED: parse_rows\n  toml: {toml_text!r}\n"
                    f"  got:  {got}\n  want: {want}",
                    file=sys.stderr,
                )
                raise SystemExit(2)

    # classify(): the two failure verdicts must not collapse into one. A
    # missing feature is an INVALID row; a compile error is an INSUFFICIENT
    # one, and only the second is evidence about the target's own code.
    class P:
        def __init__(self, rc: int, err: str) -> None:
            self.returncode, self.stderr, self.stdout = rc, err, ""

    checks = [
        (P(0, ""), BUILDS),
        (P(101, "error[E0432]: unresolved import `x`\n"), FAILS),
        (
            P(
                101,
                "error: none of the selected packages contains these features: nope\n",
            ),
            NO_FEATURE,
        ),
    ]
    for proc, want in checks:
        got, _ = classify(proc)  # type: ignore[arg-type]
        if got != want:
            print(
                f"SELFTEST FAILED: classify → {got}, want {want}", file=sys.stderr
            )
            raise SystemExit(2)


def main(argv: list[str]) -> int:
    selftest()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", nargs="?", help="one repo path; default = all contract repos")
    ap.add_argument("--package", help="only rows in this cargo package")
    ap.add_argument("--kind", help="comma list of test,bench,example")
    ap.add_argument("--grep", help="only rows whose target name contains this")
    ap.add_argument("--limit", type=int, help="stop after N rows per repo")
    ap.add_argument(
        "--target-dir",
        help="CARGO_TARGET_DIR (use /tmp/... when a sibling session is building)",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="per-row seconds")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument(
        "--list", action="store_true", help="print the rows and exit — no builds"
    )
    args = ap.parse_args(argv)

    if args.repo:
        repos = [Path(args.repo).resolve()]
    else:
        repos = derive_repos(Path(__file__).resolve().parent.parent.parent)

    if args.list:
        total = 0
        for repo in repos:
            for row in parse_rows(repo):
                print(f"{row.label}  features={','.join(row.features)}")
                total += 1
        print(f"\n{total} row(s) with required-features over {len(repos)} repo(s)")
        return 0

    reports: list[RepoReport] = []
    for repo in repos:
        if concurrent_cargo(repo):
            print(
                f"NOTE: another cargo process is working in {repo.name}/target — "
                f"pass --target-dir /tmp/<name> or a verdict here may be the box, "
                f"not the row",
                file=sys.stderr,
            )
        print(f"── {repo.name} ──", flush=True)
        reports.append(audit(repo, args))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "repo": r.repo,
                        "rows": len(r.rows),
                        "results": [
                            {
                                "label": x.row.label,
                                "verdict": x.verdict,
                                "seconds": round(x.seconds, 1),
                                "features": x.row.features,
                                "detail": x.detail,
                            }
                            for x in r.results
                        ],
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
        return 0

    print()
    print(f"{'repo':<24} {'rows':>5} {'checked':>8} {'BUILDS':>7} {'FAILS':>6} {'NO-FEAT':>8}")
    for r in reports:
        print(
            f"{r.repo:<24} {len(r.rows):>5} {len(r.results):>8} "
            f"{r.count(BUILDS):>7} {r.count(FAILS):>6} {r.count(NO_FEATURE):>8}"
        )
    bad = [x for r in reports for x in r.results if x.verdict in (FAILS, NO_FEATURE)]
    if bad:
        print(f"\n{len(bad)} row(s) that CANNOT build at their own required-features:")
        for x in bad:
            print(f"  {x.verdict:<15} {x.row.label}")
            print(f"      required-features = {x.row.features}")
            print(f"      {x.detail}")
            print(f"      {x.row.path}")
    else:
        print("\nno row failed to build at its own required-features")
    # Report, not a gate — same discipline as its siblings.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
