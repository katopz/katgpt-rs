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
UNSEEN = "UNSEEN"
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
    # The package's own declared features and dependency names, carried so the
    # free static pass can rule on a row without invoking cargo.
    feats: set[str] = field(default_factory=set)
    deps: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f"{self.repo}:{self.package}:{self.kind}:{self.name}"


@dataclass
class Group:
    """Rows sharing one package AND one EXACT feature set.

    Exact, never subset-covering. Building target T at a superset S of its own
    row and seeing it succeed proves nothing about the row — the extra
    features may be supplying the very import the row forgot, which is the
    failure this whole report exists to catch (`--all-features` builds every
    wrong row). Grouping on equality is the only sound batching: every row in
    a group is checked at literally its own feature set, in one cargo
    invocation instead of N.
    """

    package: str
    features: tuple[str, ...]
    kinds: tuple[str, ...]
    rows: list[Row]


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
        feats, deps = declared_features(data)
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
                        feats=feats,
                        deps=deps,
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


def declared_features(data: dict) -> tuple[set[str], set[str]]:
    """(features this package can enable, dependency names it can route through).

    The second set exists because `required-features` accepts `dep/feat` and
    `dep?/feat`, which name a DEPENDENCY's feature, not one of ours. That was
    measured, not assumed (cargo 1.98.1, `/tmp` probe with a `compile_error!`
    canary in the target): a `dep/extra` row is satisfied by `--features
    <ours-that-enables-it>`, by `--features dep/extra` directly, and by
    `--all-features`; only a plain no-features build skips it, correctly.
    Modelling those rows as invalid would have filed **10 riir-ai benches**
    as dead targets that are not dead.
    """
    feats = set((data.get("features") or {}).keys())
    deps: set[str] = set()

    def scan(table: dict) -> None:
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, spec in (table.get(section) or {}).items():
                deps.add(name)
                if isinstance(spec, dict):
                    if spec.get("optional"):
                        # An optional dep implies a same-named feature unless
                        # some feature already claims the name via `dep:`.
                        feats.add(name)
                    renamed = spec.get("package")
                    if renamed:
                        deps.add(renamed)

    scan(data)
    for target_table in (data.get("target") or {}).values():
        if isinstance(target_table, dict):
            scan(target_table)
    return feats, deps


def static_invalid(feature: str, feats: set[str], deps: set[str]) -> str:
    """"" if the row's entry is nameable; else why it is not.

    Free, over the whole population, and it decides the one verdict that needs
    no compiler: a row naming a feature the package does not define can never
    be satisfied by any invocation.
    """
    if "/" in feature:
        dep, _, _sub = feature.partition("/")
        dep = dep.removesuffix("?")
        if dep in deps or dep in feats:
            return ""
        return f"no dependency named `{dep}`"
    if feature.startswith("dep:"):
        # `dep:` is feature-table syntax; cargo does not accept it here.
        return "`dep:` syntax is not valid in required-features"
    if feature in feats:
        return ""
    return f"package declares no feature `{feature}`"


def group_rows(rows: list[Row]) -> list[Group]:
    """Collapse rows to (package, EXACT feature set) groups, order-preserving."""
    index: dict[tuple[str, tuple[str, ...]], Group] = {}
    for row in rows:
        key = (row.package, tuple(sorted(row.features)))
        g = index.get(key)
        if g is None:
            g = Group(package=row.package, features=key[1], kinds=(), rows=[])
            index[key] = g
        g.rows.append(row)
    out = list(index.values())
    for g in out:
        g.kinds = tuple(sorted({r.kind for r in g.rows}))
    return out


def attribute(stdout: str, rows: list[Row]) -> dict[tuple[str, str], tuple[str, str]]:
    """Per-target verdicts from one batched cargo run's JSON stream.

    Three outcomes per row, and the third is the liveness sentinel this family
    has needed every time it was omitted: a row with no error AND no artifact
    is **UNSEEN**, not BUILDS. Silence is not evidence — cargo may have stopped
    at an upstream unit, or the manifest name may not match any target cargo
    chose to build. Reading silence as success is exactly the green-zero the
    whole audit family exists to refuse.
    """
    seen: set[tuple[str, str]] = set()
    errors: dict[tuple[str, str], set[str]] = {}
    upstream_failed = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = msg.get("target") or {}
        name = target.get("name")
        kinds = target.get("kind") or []
        if not name:
            continue
        reason = msg.get("reason")
        if reason == "compiler-artifact":
            for k in kinds:
                seen.add((k, name))
        elif reason == "compiler-message":
            body = msg.get("message") or {}
            if (body.get("level") or "") != "error":
                continue
            code = ((body.get("code") or {}) or {}).get("code") or ""
            if not code:
                code = (body.get("message") or "")[:60]
            for k in kinds:
                errors.setdefault((k, name), set()).add(code)
            if any(k in ("lib", "rlib", "proc-macro", "bin", "custom-build") for k in kinds):
                upstream_failed = True
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row.kind, row.name)
        if key in errors:
            out[key] = (FAILS, ",".join(sorted(errors[key])))
        elif key in seen:
            out[key] = (BUILDS, "")
        else:
            out[key] = (
                UNSEEN,
                "an upstream unit failed first" if upstream_failed
                else "cargo built no such target",
            )
    return out


def check_group(
    repo: Path, group: Group, target_dir: str | None, timeout: int
) -> list[Result]:
    """One `cargo check` per (package, exact feature set) — not per row.

    Measured: 1,829 rows collapse to 1,070 groups over 9 repos (1.71x), and the
    saving is a whole dependency-graph rebuild per collapsed row, which is what
    the ~28 s/row mean is made of.
    """
    cmd = [
        "cargo", "check", "-p", group.package,
        "--features", ",".join(group.features),
        "--keep-going", "--message-format=json",
    ]
    # Name every target explicitly rather than `--tests/--benches/--examples`:
    # the plural flags build EVERY eligible target in the package at this
    # feature set, which is work the per-row path never did. Measured on
    # riir-clippy (44 rows / 25 groups, cold dirs, TWO pairs run in both
    # orders): plural flags cost +9% CPU-seconds against the per-row path;
    # naming the group's own targets instead measured -11% and -8% CPU-s.
    # Read the CPU number, not the wall one — wall FLIPPED SIGN between the
    # two orderings (-12%, then +13%) on a box with sibling builds live, and
    # 25 vs 44 cargo invocations is too small a difference to separate from
    # load at this repo's size. Verdicts were identical in all four runs.
    for row in group.rows:
        cmd += [KIND_FLAG[row.kind], row.name]
    env = dict(os.environ)
    env["CARGO_TERM_COLOR"] = "never"
    if target_dir:
        env["CARGO_TARGET_DIR"] = target_dir
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        el = time.monotonic() - t0
        return [Result(r, TIMEOUT, el, f"> {timeout}s (batch)") for r in group.rows]
    except OSError as e:
        el = time.monotonic() - t0
        return [Result(r, ERROR, el, str(e)) for r in group.rows]
    elapsed = time.monotonic() - t0
    # A row naming a feature the package does not define fails BEFORE any unit
    # is compiled, so there is nothing to attribute — the whole group is invalid.
    if proc.returncode != 0:
        verdict, detail = classify(proc)
        if verdict == NO_FEATURE:
            return [Result(r, NO_FEATURE, elapsed, detail) for r in group.rows]
    verdicts = attribute(proc.stdout, group.rows)
    results: list[Result] = []
    for row in group.rows:
        verdict, detail = verdicts[(row.kind, row.name)]
        results.append(Result(row, verdict, elapsed, detail))
    return results


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

    # The free pass first: a row naming a feature the package cannot enable is
    # decided by the manifest alone, so it must not cost a build. Over the 1,829
    # rows in the workspace this takes under a second; the compiler pass is
    # hours.
    static_bad: dict[str, Result] = {}
    for row in rows:
        reasons = [
            r
            for r in (static_invalid(f, row.feats, row.deps) for f in row.features)
            if r
        ]
        if reasons:
            static_bad[row.label] = Result(row, NO_FEATURE, 0.0, "; ".join(reasons))
    rows = [r for r in rows if r.label not in static_bad]

    def emit(res: Result, i: int, n: int, suffix: str = "") -> None:
        if args.quiet:
            return
        mark = "  " if res.verdict == BUILDS else "!!"
        print(
            f"{mark} [{i}/{n}] {res.verdict:<15} {res.seconds:6.1f}s "
            f"{res.row.label}{suffix}"
            + (f"  ({res.detail})" if res.detail else ""),
            # Progress belongs on stderr under --json, for the same reason the
            # repo header does: stdout is the machine-readable document, and a
            # long sweep still wants a live log.
            file=sys.stderr if args.json else sys.stdout,
            flush=True,
        )

    for res in static_bad.values():
        rep.results.append(res)
        emit(res, len(rep.results), len(rep.rows), "  [static]")

    if args.batch:
        groups = group_rows(rows)
        done = 0
        for gi, group in enumerate(groups, 1):
            for res in check_group(repo, group, args.target_dir, args.timeout):
                rep.results.append(res)
                done += 1
                emit(res, done, len(rows), f"  [grp {gi}/{len(groups)}]")
        return rep

    for i, row in enumerate(rows, 1):
        res = check_row(repo, row, args.target_dir, args.timeout)
        rep.results.append(res)
        emit(res, i, len(rows))
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

    # static_invalid(): the free pass. Its FALSE-POSITIVE direction is the
    # dangerous one — a `dep/feat` row read as invalid would have filed 10
    # riir-ai benches as dead targets, and the /tmp probe (cargo 1.98.1) shows
    # cargo satisfies those rows three different ways. Pinned in both
    # directions, including the renamed-dependency case.
    sv_toml = (
        '[package]\nname = "p"\n\n[dependencies]\n'
        'dep = { path = "../dep" }\n'
        'opt = { path = "../opt", optional = true }\n'
        'aliased = { package = "real-name", path = "../r" }\n\n'
        '[target."cfg(unix)".dependencies]\nunixdep = { path = "../u" }\n\n'
        '[features]\nmine = []\n'
    )
    sv_feats, sv_deps = declared_features(tomllib.loads(sv_toml))
    sv_cases = [
        ("mine", ""),                       # own feature
        ("opt", ""),                        # implicit feature of an optional dep
        ("dep/extra", ""),                  # a dependency's feature — VALID
        ("opt?/extra", ""),                 # weak-dep form — also valid
        ("unixdep/extra", ""),              # platform-table dependency
        ("real-name/extra", ""),            # renamed dependency, real name
        ("nope", "package declares no feature `nope`"),
        ("ghost/extra", "no dependency named `ghost`"),
        ("dep:opt", "`dep:` syntax is not valid in required-features"),
    ]
    for feature, want in sv_cases:
        got = static_invalid(feature, sv_feats, sv_deps)
        if got != want:
            print(
                f"SELFTEST FAILED: static_invalid({feature!r}) → {got!r}, "
                f"want {want!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)

    # group_rows(): grouping is on the EXACT set, and set equality must be
    # order-insensitive (a manifest may list the same features in any order)
    # while two DIFFERENT sets must never merge — merging would check a row at
    # a feature set that is not its own, which is the report's one invariant.
    g_rows = [
        Row("r", "p", "test", "a", ["x", "y"], "/a"),
        Row("r", "p", "test", "b", ["y", "x"], "/b"),
        Row("r", "p", "bench", "c", ["x"], "/c"),
        Row("r", "q", "test", "d", ["x", "y"], "/d"),
    ]
    got_groups = [
        (g.package, g.features, g.kinds, [r.name for r in g.rows])
        for g in group_rows(g_rows)
    ]
    want_groups = [
        ("p", ("x", "y"), ("test",), ["a", "b"]),
        ("p", ("x",), ("bench",), ["c"]),
        ("q", ("x", "y"), ("test",), ["d"]),
    ]
    if got_groups != want_groups:
        print(
            f"SELFTEST FAILED: group_rows\n  got:  {got_groups}\n"
            f"  want: {want_groups}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # attribute(): the batched path's parser. A regression here is silent in
    # the direction that reads as good news — no diagnostics parsed means no
    # FAILS — so every direction is pinned, including UNSEEN.
    def _art(kind: str, name: str) -> str:
        return json.dumps(
            {"reason": "compiler-artifact", "target": {"name": name, "kind": [kind]}}
        )

    def _err(kind: str, name: str, code: str | None, text: str = "boom") -> str:
        return json.dumps(
            {
                "reason": "compiler-message",
                "target": {"name": name, "kind": [kind]},
                "message": {
                    "level": "error",
                    "message": text,
                    "code": {"code": code} if code else None,
                },
            }
        )

    a_rows = [
        Row("r", "p", "test", "ok_t", ["f"], "/1"),
        Row("r", "p", "test", "bad_t", ["f"], "/2"),
        Row("r", "p", "bench", "gone_b", ["f"], "/3"),
        Row("r", "p", "test", "codeless_t", ["f"], "/4"),
    ]
    stream = "\n".join(
        [
            "   Compiling p v0.1.0",  # non-JSON noise must be skipped, not fatal
            _art("test", "ok_t"),
            _art("test", "bad_t"),
            _err("test", "bad_t", "E0432"),
            _err("test", "bad_t", "E0433"),
            _art("test", "codeless_t"),
            _err("test", "codeless_t", None, "cannot find macro `nope`"),
            json.dumps(
                {
                    "reason": "compiler-message",
                    "target": {"name": "ok_t", "kind": ["test"]},
                    "message": {"level": "warning", "message": "w", "code": None},
                }
            ),
            "{not json",
        ]
    )
    got_attr = {k: v for k, v in attribute(stream, a_rows).items()}
    want_attr = {
        ("test", "ok_t"): (BUILDS, ""),
        ("test", "bad_t"): (FAILS, "E0432,E0433"),
        ("bench", "gone_b"): (UNSEEN, "cargo built no such target"),
        ("test", "codeless_t"): (FAILS, "cannot find macro `nope`"),
    }
    if got_attr != want_attr:
        print(
            f"SELFTEST FAILED: attribute\n  got:  {got_attr}\n"
            f"  want: {want_attr}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # An upstream (lib) error explains the silence and must say so — an
    # artifact-less row is UNSEEN either way, but the two causes are different
    # findings: a broken lib is not a wrong row.
    up = attribute(
        "\n".join([_err("lib", "p", "E0599"), _art("lib", "p")]),
        [Row("r", "p", "test", "t", ["f"], "/1")],
    )
    if up[("test", "t")] != (UNSEEN, "an upstream unit failed first"):
        print(f"SELFTEST FAILED: attribute upstream → {up}", file=sys.stderr)
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
    ap.add_argument("--timeout", type=int, default=1800, help="per-row/per-group seconds")
    ap.add_argument(
        "--batch",
        action="store_true",
        help="one cargo run per (package, EXACT feature set) instead of per row "
        "(1,829 rows -> 1,070 groups over 9 repos; verdicts are per-target, and "
        "a row cargo never built reports UNSEEN, never BUILDS)",
    )
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
        if not args.json:
            # --json must be machine-readable on stdout: a progress header here
            # made the document unparseable (`Expecting value: line 1`).
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
    print(
        f"{'repo':<24} {'rows':>5} {'checked':>8} {'BUILDS':>7} {'FAILS':>6} "
        f"{'NO-FEAT':>8} {'UNSEEN':>7}"
    )
    for r in reports:
        print(
            f"{r.repo:<24} {len(r.rows):>5} {len(r.results):>8} "
            f"{r.count(BUILDS):>7} {r.count(FAILS):>6} {r.count(NO_FEATURE):>8} "
            f"{r.count(UNSEEN):>7}"
        )
    bad = [
        x
        for r in reports
        for x in r.results
        if x.verdict in (FAILS, NO_FEATURE, UNSEEN)
    ]
    if bad:
        print(
            f"\n{len(bad)} row(s) that did NOT build at their own "
            f"required-features (UNSEEN = no verdict, not a pass):"
        )
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
