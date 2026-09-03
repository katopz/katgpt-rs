#!/usr/bin/env python3
"""Targets that print a green `ok. 0 passed` because EVERY test is `#[ignore]`d.

Issue 713 T6 — a **second**, independent way a cargo target reports zero, found
by the T2b sweep rather than by the cfg auditor. `test_120_vpd_arena_goat` runs
under its features and prints

    test result: ok. 0 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out

That is not the Issue 713 shape (its `#![cfg]` is satisfied; the binary is not
empty) and `required-features` cannot address it. The reader-facing output is
nevertheless the same lie: a green `ok` over zero executed assertions, on a
target named `_goat`.

## This is a REPORT, and the reason is sharper than usual

`#[ignore]` is the **correct** marker for a slow, hardware-gated or
manual-only test. A gate here would be wrong, and folding this into Issue 713's
SILENT-NOW count would be worse — it would inflate a number whose whole value
is that every member of it is fixable by a three-line manifest row.

The distinction worth measuring is between a target with *some* ignored tests
(normal, healthy) and one where **every** test is ignored, so the binary can
never report anything but zero no matter who invokes it or how. Only the second
is reported. Whether any given one is a defect is a judgement call for its
owner; what was missing is the list.

## Two classes, deliberately apart

- **ALL-IGNORED** — ≥1 test, and every one unconditionally `#[ignore]`d.
- **NO-TESTS** — a file under `tests/` with no test attribute at all, in a
  target whose harness is cargo's. Usually a helper module that should not be a
  target, or a `fn main` driver that should be an example or a bench. Reported
  apart because the fix is different in kind.

`harness = false` targets are EXCLUDED from NO-TESTS: a custom-harness target
legitimately has no `#[test]`, prints its own output, and its exit code is its
verdict. Including them would make this report mostly noise, which is how a
report stops being read.

## What this cannot see

Counting attributes is not parsing Rust. Specifically:

- `#[cfg_attr(<cond>, ignore)]` is counted as **conditional**, never as an
  unconditional ignore — a test ignored only under miri still runs normally.
- Macro-generated tests (`#[test_case]`, `rstest` parameterisation, and any
  `macro_rules!` that emits `#[test]`) are invisible. This UNDER-counts the
  test population, so it biases toward FALSE POSITIVES.
- Unresolvable `cfg` predicates (`not(...)`, platform gates) resolve to
  COMPILED, which keeps a test in the denominator and biases toward FALSE
  NEGATIVES.

**So the bias runs in BOTH directions**, and a one-line "this can only
over-report" claim would be wrong — the first cut of this file made exactly
that claim, before per-item `cfg` resolution existed. Every ALL-IGNORED row is
a hypothesis to check by running the target, which is cheap: the whole point is
that these targets execute nothing.
- Line comments are stripped before counting, because this repo's doc comments
  quote `#[ignore]` in prose (the T6 exemplar's own header does).
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_gated_target_audit import (  # noqa: E402
    FEATURE_IN_CFG,
    TARGET_KINDS,
    cfg_body,
    default_closure,
    derive_repos,
    is_load_bearing,
    manifests,
)

# `#[test]`, `#[tokio::test]`, `#[async_std::test]`, `#[bench]`, and the
# `#[test]`-alike attribute macros used across the workspace. Anchored at the
# attribute start so `#[should_panic]` and friends do not match.
TEST_ATTR = re.compile(
    r"^\s*#\s*\[\s*(?:[a-z_][a-z0-9_]*\s*::\s*)?(?:test|bench)\s*[\]\(]",
    re.MULTILINE,
)
# Unconditional only. `#[ignore]` and `#[ignore = "reason"]`.
IGNORE_ATTR = re.compile(r"^\s*#\s*\[\s*ignore\s*(?:=|\])", re.MULTILINE)
# The reason string, when there is one. `#[ignore]` with no reason is its own
# finding shape: nothing in the source says why it never runs.
IGNORE_REASON = re.compile(r'^\s*#\s*\[\s*ignore\s*=\s*"([^"]*)"', re.MULTILINE)
# `#[cfg_attr(miri, ignore)]` — conditional, and NOT an unconditional ignore.
COND_IGNORE = re.compile(r"^\s*#\s*\[\s*cfg_attr\s*\(.*\bignore\b", re.MULTILINE)


def active_features(whole: str | None, defaults: set[str]) -> tuple[set[str], bool]:
    """The feature set under which this target actually runs, and whether it
    is ambiguous.

    Three cases, and getting the first one wrong made this report disagree
    with cargo by exactly one test on `bench_block_diagonal_goat`:

    1. **No whole-file gate, or one already satisfied by defaults** — active is
       `defaults`, unchanged. `#![cfg(any(planar_quant, iso_quant,
       hybrid_oct_pq))]` is satisfied by `planar_quant` alone, so unioning in
       ALL THREE wrongly enables `iso_quant` and counts a per-item
       `#[cfg(feature = "iso_quant")]` test as compiled. cargo said 8, the
       first cut said 9.
    2. **An unsatisfied `all(...)` or single-feature gate** — the runner must
       enable exactly those features to run the target at all, so they are
       active. This is the armed-target case (`test_120_vpd_arena_goat`).
    3. **An unsatisfied `any(...)` gate** — WHICH feature the runner picks
       changes which per-item tests compile. Returned as ambiguous rather than
       guessed: a guess produces a confident number for a configuration nobody
       ran.
    """
    if whole is None:
        return set(defaults), False
    feats = FEATURE_IN_CFG.findall(whole)
    if not feats:
        # A platform-only whole-file gate. Nothing to add; whether it compiles
        # on THIS platform is a different axis (Issue 713 T5).
        return set(defaults), False
    compact = whole.replace(" ", "").replace("\n", "")
    is_any = "any(" in compact
    satisfied = (
        any(f in defaults for f in feats) if is_any
        else all(f in defaults for f in feats)
    )
    if satisfied:
        return set(defaults), False
    if is_any:
        return set(defaults), True
    return set(defaults) | set(feats), False


def strip_line_comments(text: str) -> str:
    """Drop `//`-prefixed lines. Doc comments quote `#[ignore]` in prose."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )


ATTR_LINE = re.compile(r"^\s*#!?\s*\[")
CFG_ATTR = re.compile(r"^\s*#\s*\[\s*cfg\s*\(")


@dataclass
class TestFn:
    """One `#[test]`, with the attribute block it belongs to."""

    ignored: bool
    cfg: str | None  # the per-item `#[cfg(...)]` body, if any
    line: int
    reason: str = ""  # the `#[ignore = "..."]` string; "" means none given

    def compiled(self, defaults: set[str]) -> bool:
        """Is this test compiled under the crate's DEFAULT features?

        Resolving this is what makes the report's count checkable against
        cargo's own output. `bench_octopus_goat` has nine `#[test]` attributes
        and cargo prints `8 ignored`, because three tests are individually
        `#[cfg(feature = ...)]`-gated and one of those features is default-off.
        A counter that cannot explain that gap cannot be trusted on the files
        nobody has run.

        Unresolvable shapes (`not(...)`, a platform predicate, an unknown
        attribute) resolve to COMPILED. That is the conservative direction: it
        keeps a test in the denominator, so it can only suppress an
        ALL-IGNORED verdict, never manufacture one.
        """
        if self.cfg is None:
            return True
        body = self.cfg
        feats = FEATURE_IN_CFG.findall(body)
        if not feats or "not(" in body.replace(" ", ""):
            return True
        if "any(" in body.replace(" ", ""):
            return any(f in defaults for f in feats)
        return all(f in defaults for f in feats)


def parse_tests(text: str) -> list[TestFn]:
    """Associate each `#[test]` with its own attribute block.

    Attribute blocks are contiguous runs of `#[...]` lines: walking out from
    the test attribute in both directions until a non-attribute line is the
    whole algorithm, and it is enough because rustfmt puts one attribute per
    line. Counting `#[ignore]` and `#[test]` occurrences file-wide instead —
    which is what the first cut did — cannot tell a 2-test/1-ignore file
    (healthy) from a file whose only compiled test is ignored (a finding).
    """
    lines = text.splitlines()
    is_attr = [bool(ATTR_LINE.match(ln)) for ln in lines]
    out: list[TestFn] = []
    for i, ln in enumerate(lines):
        if not TEST_ATTR.match(ln):
            continue
        lo = i
        while lo > 0 and is_attr[lo - 1]:
            lo -= 1
        hi = i
        while hi + 1 < len(lines) and is_attr[hi + 1]:
            hi += 1
        block = lines[lo : hi + 1]
        ignored = any(IGNORE_ATTR.match(b) for b in block)
        reason = ""
        for b in block:
            m = IGNORE_REASON.match(b)
            if m:
                reason = m.group(1)
                break
        cfg = None
        for b in block:
            if CFG_ATTR.match(b):
                cfg = cfg_body(b.replace("#[", "#![", 1))
                break
        out.append(TestFn(ignored=ignored, cfg=cfg, line=i + 1, reason=reason))
    return out


@dataclass
class Target:
    repo: str
    path: str
    tests: int  # `#[test]` attributes present in the source
    compiled: int  # of those, compiled under the crate's default features
    ignored: int  # of the compiled ones, unconditionally `#[ignore]`d
    conditional: int  # `#[cfg_attr(..., ignore)]` — ignored only sometimes
    load_bearing: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class RepoReport:
    repo: str
    scanned: int = 0
    with_tests: int = 0
    all_ignored: list[Target] = field(default_factory=list)
    no_tests: list[Target] = field(default_factory=list)
    partial: int = 0  # some-but-not-all ignored: the healthy shape
    ambiguous: int = 0  # unsatisfied any(...) whole-file gate — see below


def harnessless(manifest: Path, data: dict) -> set[Path]:
    """Absolute paths of targets declaring `harness = false`."""
    out: set[Path] = set()
    crate_dir = manifest.parent
    for kind, dirname in (("test", "tests"), ("bench", "benches")):
        for row in data.get(kind, []) or []:
            if not isinstance(row, dict) or row.get("harness", True):
                continue
            rel = row.get("path") or (
                f"{dirname}/{row['name']}.rs" if "name" in row else None
            )
            if rel:
                out.add((crate_dir / rel).resolve())
    return out


def scan_manifest(repo: Path, manifest: Path, rep: RepoReport) -> None:
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (tomllib.TOMLDecodeError, OSError):
        return
    custom = harnessless(manifest, data)
    defaults = default_closure(data.get("features", {}) or {})
    crate = manifest.parent
    for dirname, _kind in TARGET_KINDS.items():
        # `examples/` are not test targets: they have no harness and no
        # `#[test]`, so every one of them would land in NO-TESTS as noise.
        if dirname == "examples":
            continue
        d = crate / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.rs")):
            rep.scanned += 1
            try:
                text = strip_line_comments(
                    f.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
            # A whole-file `#![cfg]` is NOT a reason to skip this target, and
            # skipping it was a first-cut error that dropped the very exemplar
            # Issue 713 T6 names: `test_120_vpd_arena_goat` is armed with
            # `required-features`, so its whole-file gate is off under
            # defaults — and the T6 observation is precisely that it "runs
            # under its features and reports ok. 0 passed; 3 ignored".
            #
            # So per-item cfgs are resolved against `defaults | the whole-file
            # gate's own features`: the configuration in which the target can
            # run AT ALL is the only one where "does it execute anything?" is
            # a meaningful question. Under any other configuration the answer
            # is Issue 713's, and already counted there.
            whole = cfg_body(text)
            active, ambiguous = active_features(whole, defaults)
            if ambiguous:
                # An UNSATISFIED `any(...)` whole-file gate: which single
                # feature the runner picks changes which per-item tests
                # compile, so there is no one configuration to report. Guessing
                # one would produce a confident number for a configuration
                # nobody ran.
                rep.ambiguous += 1
                continue

            tests = parse_tests(text)
            live = [t for t in tests if t.compiled(active)]
            ign = sum(1 for t in live if t.ignored)
            t = Target(
                repo=rep.repo,
                path=str(f.relative_to(repo)),
                tests=len(tests),
                compiled=len(live),
                ignored=ign,
                conditional=len(COND_IGNORE.findall(text)),
                load_bearing=is_load_bearing(f.name),
                reasons=[t.reason for t in live if t.ignored],
            )
            if not live:
                # No test the default build compiles. Either a helper module
                # that should not be a target, or every test is individually
                # cfg'd off — both print `0 passed`, and neither is fixable by
                # `required-features`.
                if f.resolve() not in custom:
                    rep.no_tests.append(t)
                continue
            rep.with_tests += 1
            if ign == len(live):
                rep.all_ignored.append(t)
            elif ign:
                rep.partial += 1


def audit(repo: Path) -> RepoReport:
    rep = RepoReport(repo=repo.name)
    for m in manifests(repo):
        scan_manifest(repo, m, rep)
    return rep


def selftest() -> None:
    """Pin the counting shapes. Runs on EVERY invocation.

    Same reasoning as the cfg auditor's: a regex regression makes this
    recognise fewer ignores and print a confident low number, which is the
    failure it exists to catch. Both directions are pinned, because a
    regression in TEST_ATTR (too few tests seen) manufactures ALL-IGNORED
    verdicts while a regression in IGNORE_ATTR erases them.
    """
    cases = [
        ("#[test]\nfn a() {}\n", 1, 0, 0),
        ("#[ignore]\n#[test]\nfn a() {}\n", 1, 1, 0),
        ('#[test]\n#[ignore = "slow"]\nfn a() {}\n', 1, 1, 0),
        # Conditional ignore is NOT unconditional: the test still runs normally.
        ("#[test]\n#[cfg_attr(miri, ignore)]\nfn a() {}\n", 1, 0, 1),
        # Namespaced test attributes.
        ("#[tokio::test]\nfn a() {}\n", 1, 0, 0),
        ("#[async_std::test]\nfn a() {}\n", 1, 0, 0),
        ("#[bench]\nfn a(b: &mut Bencher) {}\n", 1, 0, 0),
        # Attribute macros with arguments.
        ("#[tokio::test(flavor = \"multi_thread\")]\nfn a() {}\n", 1, 0, 0),
        # Not test attributes.
        ("#[should_panic]\nfn a() {}\n", 0, 0, 0),
        ("#[allow(dead_code)]\nfn a() {}\n", 0, 0, 0),
        # `#[test]` quoted in a doc comment must not count — the T6 exemplar's
        # own header does exactly this with `--ignored`.
        ("//! run with #[ignore]\n#[test]\nfn a() {}\n", 1, 0, 0),
        ("// #[test]\nfn a() {}\n", 0, 0, 0),
    ]
    for src, want_t, want_i, want_c in cases:
        text = strip_line_comments(src)
        tests = parse_tests(text)
        got = (
            len(tests),
            sum(1 for t in tests if t.ignored),
            len(COND_IGNORE.findall(text)),
        )
        assert got == (want_t, want_i, want_c), f"{src!r} -> {got}, want {(want_t, want_i, want_c)}"

    # PER-TEST association, which a file-wide count cannot do. 2 tests / 1
    # ignore is HEALTHY; the same two counts with the ignore on the only
    # COMPILED test is a finding. A file-wide counter scores both identically.
    text = strip_line_comments("#[test]\nfn a() {}\n#[ignore]\n#[test]\nfn b() {}\n")
    tests = parse_tests(text)
    assert [t.ignored for t in tests] == [False, True], "attribute block mis-associated"

    # The ignore must attach to ITS OWN test, in either attribute order, and
    # must not leak across a `fn` boundary into the next test's block.
    text = strip_line_comments(
        '#[test]\n#[ignore = "x"]\nfn a() {}\n\n#[test]\nfn b() {}\n'
    )
    assert [t.ignored for t in parse_tests(text)] == [True, False], "ignore leaked"

    # Per-item `#[cfg]` resolved against the default closure. This is the pin
    # that makes the count checkable against cargo: tests/bench_octopus_goat.rs
    # has NINE `#[test]` attributes and cargo prints `8 ignored`, because three
    # are individually cfg-gated and one gating feature is default-off.
    text = strip_line_comments(
        '#[test]\n#[ignore]\nfn a() {}\n\n'
        '#[cfg(feature = "on")]\n#[test]\n#[ignore]\nfn b() {}\n\n'
        '#[cfg(feature = "off")]\n#[test]\n#[ignore]\nfn c() {}\n\n'
        '#[cfg(all(feature = "on", feature = "off"))]\n#[test]\nfn d() {}\n\n'
        '#[cfg(any(feature = "on", feature = "off"))]\n#[test]\n#[ignore]\nfn e() {}\n'
    )
    tests = parse_tests(text)
    assert len(tests) == 5, f"parsed {len(tests)} tests, want 5"
    live = [t for t in tests if t.compiled({"on"})]
    assert len(live) == 3, f"{len(live)} compiled, want 3 (a, b, e)"
    assert all(t.ignored for t in live), "a compiled test lost its ignore"

    # active_features' three cases. Cases 1 and 2 were each a measured
    # disagreement with cargo before this function existed, on a real file.
    #
    # Case 1: a whole-file `any(...)` SATISFIED by defaults adds nothing.
    # tests/bench_block_diagonal_goat.rs is `#![cfg(any(planar_quant,
    # iso_quant, hybrid_oct_pq))]` with planar_quant default-on; unioning all
    # three enables iso_quant and counts a per-item iso_quant test as
    # compiled. cargo prints 8 ignored; the first cut said 9.
    act, amb = active_features('any(feature = "on", feature = "off")', {"on"})
    assert act == {"on"} and not amb, f"satisfied any() over-approximated: {act}"
    # Case 2: an UNSATISFIED all()/single gate — the runner must enable it, so
    # it is active. This is the armed-target case (test_120_vpd_arena_goat).
    act, amb = active_features('feature = "opt"', set())
    assert act == {"opt"} and not amb, f"armed gate not activated: {act}"
    act, amb = active_features('all(feature = "a", feature = "b")', set())
    assert act == {"a", "b"} and not amb, f"all() gate not activated: {act}"
    # Case 3: an UNSATISFIED any() gate is AMBIGUOUS, not guessed.
    act, amb = active_features('any(feature = "a", feature = "b")', set())
    assert amb, "unsatisfied any() was resolved to a single guessed config"
    # No gate, and a platform-only gate, both leave defaults alone.
    assert active_features(None, {"d"}) == ({"d"}, False)
    assert active_features('target_os = "macos"', {"d"}) == ({"d"}, False)

    import tempfile

    # The whole-file gate's own features count as ACTIVE. Without this the
    # exemplar T6 names (an armed target, whole-file gate off under defaults)
    # is skipped entirely, which is the first-cut error this pins against.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "armed_goat.rs").write_text(
            '#![cfg(feature = "opt")]\n#[test]\n#[ignore]\nfn a() {}\n'
        )
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\nversion = "0.0.0"\n\n'
            "[features]\nopt = []\n\n"
            '[[test]]\nname = "armed_goat"\nrequired-features = ["opt"]\n'
        )
        r = RepoReport(repo="p")
        scan_manifest(root, root / "Cargo.toml", r)
        assert len(r.all_ignored) == 1, (
            "an armed, whole-file-gated target was skipped — the T6 exemplar's "
            "own shape"
        )
        assert r.all_ignored[0].load_bearing, "armed_goat read as not load-bearing"

    # Unresolvable predicates resolve to COMPILED — the direction that can only
    # SUPPRESS a verdict, never manufacture one.
    for body in ('not(feature = "on")', 'target_os = "macos"'):
        t = TestFn(ignored=False, cfg=body, line=1)
        assert t.compiled({"on"}), f"unresolvable cfg {body!r} read as compiled-out"

    # harness = false must be excluded from NO-TESTS.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "driver.rs").write_text("fn main() {}\n")
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\nversion = "0.0.0"\n\n'
            '[[test]]\nname = "driver"\nharness = false\n'
        )
        r = RepoReport(repo="p")
        scan_manifest(root, root / "Cargo.toml", r)
        assert r.no_tests == [], "a harness=false driver was reported as NO-TESTS"


def main(argv: list[str]) -> int:
    selftest()

    argv = list(argv)
    as_json = "--json" in argv
    if as_json:
        argv.remove("--json")

    if len(argv) > 1:
        repos = [Path(a).resolve() for a in argv[1:]]
        scope = "argument"
    else:
        here = Path(__file__).resolve().parent.parent
        repos = derive_repos(here.parent)
        scope = "derived (BOUNDARY.md + .git)"

    if as_json:
        # Machine-readable, for `cfg_gated_floor_gate.py`'s reasonless pin. The
        # consumer must not re-derive any of this — a second copy of the parser
        # is a second thing to keep in step.
        import json

        out = {}
        for r in (audit(x) for x in repos):
            reasonless = [
                t.path for t in r.all_ignored if any(not x.strip() for x in t.reasons)
            ]
            out[r.repo] = {
                "scanned": r.scanned,
                "with_tests": r.with_tests,
                "all_ignored": len(r.all_ignored),
                "all_ignored_load_bearing": sum(
                    1 for t in r.all_ignored if t.load_bearing
                ),
                # The PATHS, not just the count. A count is not a checksum over
                # a set: one justified ALL-IGNORED target removed and one
                # unjustified one added leaves the total unmoved, which is
                # exactly how katgpt-rs went 3 -> 5 load-bearing with every
                # number in sight agreeing. `cfg_gated_floor_gate.py` pins the
                # MEMBERSHIP against scripts/all_ignored_load_bearing.txt.
                "all_ignored_load_bearing_paths": sorted(
                    t.path for t in r.all_ignored if t.load_bearing
                ),
                "reasonless_targets": len(reasonless),
                "reasonless_paths": sorted(reasonless),
                "partial": r.partial,
                "no_tests": len(r.no_tests),
                "ambiguous": r.ambiguous,
            }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"all-ignored target audit — {len(repos)} repo(s), population {scope}\n")
    header = (
        f"{'repo':<24} {'targets':>8} {'w/ tests':>9} {'ALL-IGNORED':>12} "
        f"{'load-bear':>10} {'partial':>8} {'no-tests':>9} {'ambig':>6}"
    )
    print(header)
    print("-" * len(header))

    reports = [audit(r) for r in repos]
    tot_all = tot_lb = tot_no = 0
    for rep in reports:
        lb = sum(1 for t in rep.all_ignored if t.load_bearing)
        tot_all += len(rep.all_ignored)
        tot_lb += lb
        tot_no += len(rep.no_tests)
        print(
            f"{rep.repo:<24} {rep.scanned:>8} {rep.with_tests:>9} "
            f"{len(rep.all_ignored):>12} {lb:>10} {rep.partial:>8} "
            f"{len(rep.no_tests):>9} {rep.ambiguous:>6}"
        )

    print(
        f"\nALL-IGNORED {tot_all} ({tot_lb} load-bearing by name): the binary compiles,\n"
        f"runs, and can NEVER print anything but `ok. 0 passed` — every test in it is\n"
        f"unconditionally #[ignore]d. `required-features` cannot address this shape,\n"
        f"so it is NOT part of Issue 713's SILENT-NOW count. NO-TESTS {tot_no}: a file\n"
        f"under tests/ with no test attribute and cargo's own harness.\n"
    )
    for rep in reports:
        if not rep.all_ignored:
            continue
        print(f"  {rep.repo}")
        for t in sorted(rep.all_ignored, key=lambda x: (not x.load_bearing, x.path)):
            mark = "  [LOAD-BEARING]" if t.load_bearing else ""
            cond = f", {t.conditional} conditional" if t.conditional else ""
            print(f"    {t.path}{mark}")
            print(
                f"      {t.tests} test(s) in source, {t.compiled} compiled under "
                f"the running config, {t.ignored} unconditionally ignored{cond}"
            )
        print()

    if tot_all == 0:
        print("  (none)\n")

    # Reasons, because a count is not a diagnosis. "requires a GPU" and "slow,
    # run with --release --ignored" are legitimate and self-documenting; an
    # EMPTY reason is the one shape that is a finding on its own terms — the
    # source says nothing about why the test never runs, so no reader can tell
    # a deliberate manual-only test from one parked during a refactor and
    # forgotten.
    from collections import Counter

    reasons: Counter[str] = Counter()
    for rep in reports:
        for t in rep.all_ignored:
            for r in t.reasons:
                reasons[r.strip() or "(NO REASON GIVEN)"] += 1
    if reasons:
        blank = reasons.get("(NO REASON GIVEN)", 0)
        total_r = sum(reasons.values())
        print(
            f"  Why they are ignored — {len(reasons)} distinct reason(s) over "
            f"{total_r} ignored test(s), {blank} with NO reason string:\n"
        )
        for reason, n in reasons.most_common(12):
            short = reason if len(reason) <= 88 else reason[:85] + "..."
            print(f"    {n:>5}  {short}")
        if len(reasons) > 12:
            print(f"    {'':>5}  … {len(reasons) - 12} more")
        print()

    print(
        "Report, not a gate — exit 0 always, and here that is a stronger claim than\n"
        "usual: #[ignore] is the CORRECT marker for a slow or hardware-gated test.\n"
        "Every ALL-IGNORED row is a hypothesis for its owner, not a defect, and\n"
        "the bias runs BOTH ways: macro-generated tests are invisible (toward false\n"
        "positives) and unresolvable cfg predicates count as compiled (toward false\n"
        "negatives). Check a row by running the target — it executes nothing, so it\n"
        "is the cheapest verification there is."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
