#!/usr/bin/env python3
"""Per-feature isolation gate, bounded by the diff instead of the manifest.

`--all-features` proves the UNION compiles. It says nothing about a feature
compiling ON ITS OWN, and every primitive here ships behind an opt-in flag, so
"feature X alone is broken" is a live failure mode the full gate is blind to by
construction (Issue 701 R1).

The standard tool, `cargo hack --each-feature`, is not affordable here. Measured
2026-09-01 on an M3 Max with a warm target dir: mean 39.5s per flag over a
seeded random sample of 6 (range 4.2s-110.2s), so 568 flags is ~6.2 h and even
the 197 default-on ones are ~2.2 h. Marginal disk was ~0.09 GiB per flag against
60 GiB free.

So: check only the flags whose DEFINITION the diff touches. A typical 1-3 flag
change costs ~40-120s, which is affordable per-PR, and it catches the case that
actually regresses — someone adding or editing a flag without ever building it
alone.

    scripts/feature_isolation_gate.py [base_ref]

base_ref defaults to $GITHUB_BASE_REF (set on GitHub PRs), then origin/develop.
Exit 0 = every touched flag builds alone, or none were touched. Exit 1 = a flag
does not build in isolation, or the gate could not determine what to check.
"""

from __future__ import annotations

import os
import random
import re
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

# Same directory, so a plain import resolves. Reused rather than re-derived on
# purpose: "which flags are default-on" is subtle (a `default` entry may be a
# `pkg/flag` passthrough naming a flag this manifest does not define), and two
# definitions of it would drift. count_features.py is also what Issue 701's
# "197 default-on" figure comes from, so the scope below is that number by
# construction instead of by coincidence.
from count_features import load_features
from bench_doc_audit import iter_cargo_manifests

ROOT = Path(__file__).resolve().parent.parent

# In CI an ambiguous signal must fail; locally the same signal is usually just
# "nothing changed". Both GitHub and most other runners set one of these.
IN_CI = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"

# `foo = [...]` at the start of a line inside a [features] table. Cargo feature
# names are [a-zA-Z0-9_-].
FEATURE_DEF_RE = re.compile(r"^\+([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*\[")
DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.*)$")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def rev(ref: str) -> str | None:
    r = run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    return r.stdout.strip() or None


def resolve_base(argv: list[str]) -> tuple[str | None, str | None]:
    """(ref, error). Prefers the REMOTE-tracking ref over a same-named local.

    $GITHUB_BASE_REF is a bare branch name ("develop"). Resolving that to the
    LOCAL branch is wrong in CI and was actively dangerous: in a shallow
    single-branch clone the local `develop` IS HEAD, so the gate diffed HEAD
    against itself, found no touched flags, and reported a green pass. That is
    the vacuous green this gate exists to prevent, produced by the gate itself.
    """
    head = rev("HEAD")

    # An explicitly requested base that does not resolve is an error, not a cue
    # to fall back. Silently substituting origin/develop for a typo means the
    # gate reports on a base the caller never asked for — and the diagnostic
    # then names the substitute, not the typo.
    explicit = argv[1] if len(argv) > 1 else None
    if explicit and not any(rev(r) for r in
                            ([f"origin/{explicit}", explicit]
                             if "/" not in explicit else [explicit])):
        return None, f"explicit base ref {explicit!r} does not resolve"

    for cand in (explicit,
                 os.environ.get("GITHUB_BASE_REF"),
                 "develop"):
        if not cand:
            continue
        # remote first, then the literal ref as given
        for ref in ([f"origin/{cand}", cand] if "/" not in cand else [cand]):
            sha = rev(ref)
            if sha is None:
                continue
            if sha == head:
                # base == HEAD means the diff is empty, which is AMBIGUOUS:
                #
                #   locally on an up-to-date develop -> genuinely nothing to
                #     compare, and exit 1 would be obstructive noise;
                #   in CI on a PR -> the base was never fetched, and an empty
                #     diff is the vacuous green this gate exists to prevent.
                #
                # Same observation, opposite correct response, so it is resolved
                # by context rather than by picking one and being wrong half the
                # time.
                if IN_CI:
                    return None, (
                        f"base {ref!r} resolves to HEAD ({sha[:12]}) — in CI "
                        f"that means the base was not fetched, so an empty diff "
                        f"proves nothing. Use fetch-depth: 0, or pass an "
                        f"explicit base ref.")
                return None, (
                    f"__NOOP__base {ref!r} == HEAD ({sha[:12]}); nothing to "
                    f"compare (local up-to-date checkout)")
            if run(["git", "merge-base", ref, "HEAD"]).returncode != 0:
                return None, (
                    f"no merge base between {ref!r} and HEAD — almost always a "
                    f"shallow clone. Use fetch-depth: 0.")
            return ref, None
    return None, ("no usable base ref (tried argv, $GITHUB_BASE_REF, "
                  "origin/develop, develop)")


def manifest_facts(cargo_toml: Path) -> tuple[str | None, set[str]]:
    """(package name, declared feature names) for a manifest at HEAD."""
    try:
        with cargo_toml.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None, set()
    feats = data.get("features", {})
    return (data.get("package", {}).get("name"),
            {k for k in feats if k != "default"})


def changed_flags(base: str) -> list[tuple[str, str]]:
    """[(package, flag)] for every feature DEFINITION added/edited vs base."""
    diff = run(["git", "diff", "--unified=0", f"{base}...HEAD", "--",
                "*Cargo.toml"])
    if diff.returncode != 0:
        print(f"  ✗ git diff against {base} failed: {diff.stderr.strip()}")
        sys.exit(1)
    out: list[tuple[str, str]] = []
    cur_pkg: str | None = None
    cur_feats: set[str] = set()
    for line in diff.stdout.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            cur_pkg, cur_feats = manifest_facts(ROOT / m.group(1))
            continue
        m = FEATURE_DEF_RE.match(line)
        if not (m and cur_pkg):
            continue
        flag = m.group(1)
        # `name = [...]` is not unique to [features]. `required-features = [..]`
        # inside [[bench]] / [[test]] has the identical shape, and with
        # --unified=0 the diff never shows the enclosing table header, so the
        # section CANNOT be inferred from the diff text. The gate's first canary
        # duly reported `katgpt-core/required-features` as a broken feature.
        #
        # So candidates are confirmed against the manifest's real [features]
        # table at HEAD. This also correctly drops a flag whose definition the
        # diff DELETED: absent from HEAD's features, nothing to isolate.
        if flag == "default" or flag not in cur_feats:
            continue
        if (cur_pkg, flag) not in out:
            out.append((cur_pkg, flag))
    return out


def collect_flags(scope: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(in-scope, passthrough-only) (package, flag) pairs for a named scope.

    scope="default-on" -> only flags in a manifest's `default` array.
    scope="all"        -> every flag the manifest declares (Issue 701's 568).

    A manifest's `default` array may name `otherpkg/flag`. Such an entry is
    default-on *as this package consumes it*, but the flag is defined
    elsewhere and `cargo check -p thispkg --features flag` would not resolve
    it — so it cannot be isolated here. Those are returned SEPARATELY and
    reported, never silently dropped ("no silent caps"): they are a real gap
    in this scope, not an absence of one.
    """
    inscope: list[tuple[str, str]] = []
    passthrough: list[tuple[str, str]] = []
    for toml in sorted(iter_cargo_manifests(ROOT)):
        default_on, all_flags, _ = load_features(toml)
        if not all_flags:
            continue
        try:
            with toml.open("rb") as f:
                pkg = tomllib.load(f).get("package", {}).get("name")
        except Exception:
            continue
        if not pkg:
            continue
        wanted = all_flags if scope == "all" else default_on
        for flag in sorted(wanted):
            (inscope if flag in all_flags else passthrough).append((pkg, flag))
    return inscope, passthrough


def check_flags(targets: list[tuple[str, str]]) -> tuple[list, list[float]]:
    """Build each (package, flag) alone.

    Returns (failures, per-flag seconds, (pkg, flag, seconds) rows).

    Timing is collected because Issue 701 R1b is blocked on a real measurement,
    not on an estimate: the figure it carries is an extrapolation from n=6 with
    a 26x range, which is a point estimate wearing the costume of a bound."""
    failed, secs, rows = [], [], []
    for i, (pkg, flag) in enumerate(targets, 1):
        cmd = ["cargo", "check", "-p", pkg, "--no-default-features",
               "--features", flag]
        t0 = time.monotonic()
        r = run(cmd)
        dt = time.monotonic() - t0
        secs.append(dt)
        rows.append((pkg, flag, dt))
        if r.returncode == 0:
            print(f"  [{i}/{len(targets)}] ✓ {pkg}/{flag}  {dt:.1f}s", flush=True)
        else:
            failed.append((pkg, flag))
            tail = [l for l in r.stderr.splitlines()
                    if l.startswith("error")][:5]
            print(f"  [{i}/{len(targets)}] ✗ {pkg}/{flag} does NOT build alone"
                  f"  {dt:.1f}s", flush=True)
            for l in tail:
                print(f"      {l}", flush=True)
    return failed, secs, rows


def selftest() -> None:
    """Pin the scope invariant. Cheap, no fixtures, and it can actually fail.

    `all` must be a superset of `default-on` — they walk the same manifests and
    differ only in which flag names they keep. If a future edit filters the two
    scopes differently (say, skipping a manifest in one path), the isolation
    sweep would silently under-cover and still print a confident pass, which is
    the failure mode this whole gate exists to prevent."""
    d, _ = collect_flags("default-on")
    a, _ = collect_flags("all")
    missing = set(d) - set(a)
    if missing:
        raise SystemExit(
            "✗ scope self-test FAILED — default-on is not a subset of all\n"
            f"  {len(missing)} pair(s) missing, e.g. {sorted(missing)[:3]}")
    if not d or not a:
        raise SystemExit(
            f"✗ scope self-test FAILED — empty scope "
            f"(default-on={len(d)}, all={len(a)}); the manifest walk found "
            f"nothing, so any sweep would pass vacuously")


def parse_opts(argv: list[str]) -> tuple[dict, list[str]]:
    """Pull the scope options out, leave the positional base ref in place.

    Hand-parsed rather than argparse'd to keep the existing calling convention
    exactly: `feature_isolation_gate.py [base_ref]` is what
    .github/workflows/feature_isolation.yml passes today, and a gate is not the
    place to discover that its own invocation changed shape."""
    opts = {"scope": "diff", "sample": 0, "seed": 701, "list": False}
    rest, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--scope" and i + 1 < len(argv):
            opts["scope"] = argv[i + 1]; i += 2; continue
        if a == "--sample" and i + 1 < len(argv):
            opts["sample"] = int(argv[i + 1]); i += 2; continue
        if a == "--seed" and i + 1 < len(argv):
            opts["seed"] = int(argv[i + 1]); i += 2; continue
        if a == "--list":
            opts["list"] = True; i += 1; continue
        rest.append(a); i += 1
    return opts, rest


def report_by_package(rows: list[tuple[str, str, float]]) -> None:
    """Per-package cost. The global mean hides the only variable that matters.

    Measured 2026-09-01: a sweep's per-flag cost is dominated by how much the
    PREVIOUS check left warm, not by the flag. Consecutive
    `--no-default-features` builds share almost the whole dependency graph, so
    only the top crate rebuilds — which is why Issue 701's 39.5s/flag figure
    (measured as isolated checks against a default-featured target dir, i.e.
    the per-PR cost) over-estimates a SWEEP by an order of magnitude."""
    by: dict[str, list[float]] = {}
    for pkg, _flag, dt in rows:
        by.setdefault(pkg, []).append(dt)
    print("\n▸ per package (n, mean, total):")
    for pkg in sorted(by, key=lambda k: -sum(by[k])):
        v = by[pkg]
        print(f"    {pkg:<22} n={len(v):<4} mean {statistics.fmean(v):6.1f}s"
              f"   total {sum(v) / 60:5.1f} min")


def report_timing(secs: list[float], total_scope: int) -> None:
    """The measurement Issue 701 R1b is blocked on, printed as a range."""
    if not secs:
        return
    mean, med = statistics.fmean(secs), statistics.median(secs)
    print(f"\n▸ timing over {len(secs)} flag(s): "
          f"mean {mean:.1f}s  median {med:.1f}s  "
          f"min {min(secs):.1f}s  max {max(secs):.1f}s  "
          f"total {sum(secs) / 60:.1f} min")
    if len(secs) < total_scope:
        # Both, because they differ a lot on this distribution and quoting only
        # the mean is how the n=6 estimate in Issue 701 came to be read as a
        # bound. A long tail makes the mean the honest planning number and the
        # median the honest typical-flag number; neither alone is the answer.
        print(f"  extrapolated to all {total_scope}: "
              f"{mean * total_scope / 3600:.1f} h (mean) / "
              f"{med * total_scope / 3600:.1f} h (median)")
        print(f"  point estimate from n={len(secs)}, NOT a bound — "
              f"the per-flag range here is "
              f"{max(secs) / max(min(secs), 0.1):.0f}x")


def run_scope(opts: dict) -> int:
    targets, passthrough = collect_flags(opts["scope"])
    total = len(targets)
    names = len({f for _, f in targets})
    print(f"▸ scope: {opts['scope']} — {total} (package, flag) pair(s) "
          f"across {names} unique flag name(s)")
    if total != names:
        # Issue 701 sizes this work as "197 default-on flags". That is the
        # unique-NAME count; the same name defined in two manifests is two
        # different builds and can pass in one and fail in the other.
        print(f"  note: {total - names} pair(s) are a flag name defined in "
              f"more than one manifest — each is its own build")
    if passthrough:
        print(f"  note: {len(passthrough)} default entry(ies) are "
              f"`pkg/flag` passthroughs, not isolable from here")
    if opts["sample"] and opts["sample"] < total:
        rng = random.Random(opts["seed"])
        targets = sorted(rng.sample(targets, opts["sample"]))
        print(f"▸ sampled {len(targets)} of {total} (seed {opts['seed']}, "
              f"reproducible)")
    if opts["list"]:
        for pkg, flag in targets:
            print(f"    {pkg}/{flag}")
        return 0
    failed, secs, rows = check_flags(targets)
    report_by_package(rows)
    report_timing(secs, total)
    if failed:
        print(f"\n✗ feature isolation FAILED — {len(failed)}/{len(targets)}: "
              + ", ".join(f"{p}/{f}" for p, f in failed))
        return 1
    print(f"\n✓ feature isolation PASSED — {len(targets)}/{len(targets)} "
          f"flag(s) build alone")
    return 0


def main(argv: list[str]) -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    selftest()
    opts, argv = parse_opts(argv)
    if opts["scope"] != "diff":
        if opts["scope"] not in ("default-on", "all"):
            print(f"✗ unknown --scope {opts['scope']!r} "
                  f"(known: diff, default-on, all)")
            return 1
        return run_scope(opts)
    base, err = resolve_base(argv)
    if base is None:
        if err and err.startswith("__NOOP__"):
            print(f"✓ {err[len('__NOOP__'):]}")
            return 0
        # Not a skip: without a trustworthy base we do not know what changed,
        # and reporting a pass would be a claim we cannot support.
        print(f"✗ {err}")
        return 1
    print(f"▸ base: {base}")

    targets = changed_flags(base)
    if not targets:
        # Say so explicitly. A silent pass here reads identical to a real one.
        print("✓ no feature DEFINITION touched vs base — nothing to isolate")
        return 0

    print(f"▸ {len(targets)} touched flag(s): "
          + ", ".join(f"{p}/{f}" for p, f in targets))
    failed, _, _ = check_flags(targets)
    if failed:
        print(f"✗ feature isolation FAILED — {len(failed)}/{len(targets)}: "
              + ", ".join(f"{p}/{f}" for p, f in failed))
        return 1
    print(f"✓ feature isolation PASSED — {len(targets)}/{len(targets)} "
          f"flag(s) build alone")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
