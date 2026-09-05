#!/usr/bin/env python3
"""Issue 513 — does a target's `required-features` row SATISFY its own `#![cfg]`?

The compiler-verified sweep (`required_features_build_audit.py`) answers "does
this row build its target". It cannot answer this one, and the difference is a
whole failure class:

    #![cfg(feature = "dflare_training")]        // the file
    required-features = ["dflash_training"]     // the row

Both features exist. `dflare_training` is the STRICTLY STRONGER one (it enables
`dflash_training`). At its own row the file compiles to NOTHING, the harness
prints `running 0 tests / test result: ok. 0 passed`, and cargo exits 0 — so
the build audit reports **BUILDS** and is right. Every other instrument agrees
too: the row exists, so `cfg_gated_target_audit.py` counts the reader as
protected. Measured 2026-09-06, riir-train `054a39a2`: fixing the row took that
target from 0 passed to 1 passed, an assertion that had never executed at any
revision.

So the three shapes this issue now covers are:

    the row cannot build its target        -> FAILS-TO-BUILD   (compiler)
    the LIBRARY cannot build at the row    -> UNSEEN           (compiler)
    the row BUILDS and compiles to NOTHING -> BUILDS           (THIS SCRIPT)

The third is the worst to hold, because nothing reds.

## What it computes

For each row, the leading inner `#![cfg(...)]` attributes of its target source
(multiple ones AND together), reduced to the set of feature names they require.
Then the row's feature CLOSURE **within its own package** — following
`feat = [...]` entries, and deliberately NOT following `dep/feat`, because
enabling a dependency's feature does not enable one of ours. A feature the cfg
requires and the closure lacks means the target is empty at its own row.

## DEFAULT features are part of the resolved set — the trap this hit first

Cargo checks a `required-features` row against the **resolved** feature set,
and an ordinary `cargo test --features X` resolves `default` too. A first cut
computed the closure of the ROW ALONE and reported **6 riir-mmorpg-examples
targets as EMPTY-AT-ROW** — every one a false positive, because their
`feature = "authority"` is in that package's `default` list. Caught by
EXECUTING one of them (3 passed at its bare row, not 0), not by re-reading the
model.

So the closure is seeded with the row AND `default`. `EMPTY-WITHOUT-DEFAULTS`
is reported as its own, weaker class: real (that target IS empty under
`--no-default-features`) but not a defect, because no row can be expected to
re-name a default.

## Non-feature conjuncts are STRIPPED, not surrendered to

A first cut treated any predicate mentioning `test` / `target_os` /
`debug_assertions` as UNRESOLVED, and that put **137 of 1,174 rows** — 12% —
into the bucket where findings hide. The shapes were dull and dominant:
`all(test, feature = "X")` (32), `all(feature = "X", not(target_os = "X"))`
(10), plus 48 bare `cfg(test)`.

They do not need to be there. The question is "does the ROW enable the features
this file requires", and a platform or profile conjunct is INDEPENDENT of it: if
`feature = "X"` is missing, the target is empty at its own row whether or not
the platform conjunct also holds. So a top-level conjunction is split, the
non-feature atoms are dropped, and the verdict is made on what remains. A
predicate with no feature atom at all is `NO-FEATURE-REQ` — real, and not the
row's fault.

## What it still declines to rule on

`feature` appearing under `any(...)` or `not(...)` stays UNRESOLVED, and those
are genuinely undecidable here rather than merely awkward: `any` is the shape
cargo's AND-only `required-features` cannot express at all, and under `not` the
implication INVERTS — enabling the feature is what would empty the target. Read
the UNRESOLVED count; an audit's unresolved bucket is where findings hide, and
this family has already paid for treating one as empty.

A **report, not a gate** (always exit 0) for the reason its siblings are: the
population spans repos whose owners have not taken this issue. The gateable
half is the katgpt-rs slice, and it is gateable precisely because it needs no
compiler.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg_gated_target_audit import derive_repos  # noqa: E402
from required_features_build_audit import Row, parse_rows  # noqa: E402

INNER_CFG = re.compile(r"^\s*#!\[cfg\((.*)\)\]\s*$")
FEATURE = re.compile(r'feature\s*=\s*"([^"]+)"')

# Predicates that are not about cargo features. Reducing one of these to "no
# features required" would be a silent pass, so they make the whole target
# UNRESOLVED instead.
NON_FEATURE = ("target_os", "target_arch", "target_family", "target_env",
               "target_pointer_width", "target_vendor", "debug_assertions",
               "miri", "test", "doc", "doctest", "windows", "unix")

EMPTY = "EMPTY-AT-ROW"
EMPTY_NODEFAULT = "EMPTY-WITHOUT-DEFAULTS"
SATISFIED = "SATISFIED"
UNRESOLVED = "UNRESOLVED"
NO_FEATURE_REQ = "NO-FEATURE-REQ"
NO_CFG = "NO-CFG"


def split_conjuncts(pred: str) -> list[str]:
    """Top-level atoms of a cfg predicate, unwrapping one layer of `all(...)`.

    Splits on commas at paren depth 0 so a nested `not(target_os = "x")` stays
    one atom. A bare predicate is its own single conjunct.
    """
    p = pred.strip()
    if p.startswith("all(") and p.endswith(")"):
        p = p[4:-1]
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in p:
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return [a for a in out if a]


@dataclass
class Finding:
    row: Row
    verdict: str
    needs: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    why: str = ""


def leading_inner_cfgs(path: str) -> list[str] | None:
    """The `#![cfg(...)]` predicates before the first item, in file order.

    None means the file could not be read — distinct from [] (read fine, no
    inner cfg), because "no cfg" is a verdict and "cannot read" is not.

    Only INNER attributes (`#![...]`) count, and only before the first item:
    an inner attribute after an item is a compile error, and an OUTER
    `#[cfg]` gates one item rather than the file. Doc comments, `#![allow]`
    and friends are skipped rather than terminating the scan.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*")):
            continue
        if stripped.startswith("#!["):
            m = INNER_CFG.match(line)
            if m:
                out.append(m.group(1))
            continue
        break  # first real item — inner attributes cannot follow it
    return out


def required_features(pred: str) -> set[str] | None:
    """The feature names a predicate REQUIRES, or None if undecidable here.

    An empty set means "decidable, and requires no feature" — a platform- or
    profile-only cfg, which is real and is not the row's fault. That is why the
    return distinguishes `set()` from `None`; collapsing them would either
    invent a finding or hide one.
    """
    needs: set[str] = set()
    for atom in split_conjuncts(pred):
        names = FEATURE.findall(atom)
        if not names:
            # A platform / profile / test atom. Independent of the feature
            # question: a missing feature empties the target whether or not
            # this atom holds. Drop it rather than surrendering the whole row.
            if any(tok in atom for tok in NON_FEATURE):
                continue
            return None  # an atom we do not model at all — do not guess
        if atom.startswith(("any(", "not(")) or "any(" in atom or "not(" in atom:
            # `any` is the shape cargo's AND-only required-features cannot
            # express; under `not` the implication INVERTS. Neither is a
            # requirement this report can rule on.
            return None
        needs |= set(names)
    return needs


def feature_map(crate_dir: Path) -> dict[str, list[str]]:
    try:
        data = tomllib.loads(
            (crate_dir / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
        )
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {k: (v or []) for k, v in (data.get("features") or {}).items()}


def default_features(fmap: dict[str, list[str]]) -> list[str]:
    """The package's `default` list — part of every ordinary resolve.

    Not an optional refinement: omitting it made this report emit six false
    EMPTY-AT-ROW findings in one repo, all on a feature that is simply ON by
    default. See the module docstring.
    """
    return list(fmap.get("default", []))


def closure(fmap: dict[str, list[str]], start: list[str]) -> set[str]:
    """Feature names enabled IN THIS PACKAGE by `start`.

    `dep/feat` entries are deliberately not followed: enabling a dependency's
    feature does not enable one of ours, and treating it as if it did is
    exactly the confusion that produced instance 7 (`self_advantage_gate_bench`
    forwards `katgpt-core/sense_composition` AND our `sense_composition`, and
    satisfies neither `#![cfg(feature = "sense_composition_bench")]`).
    """
    seen: set[str] = set()
    stack = list(start)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for nxt in fmap.get(f, []):
            if "/" not in nxt and nxt not in seen:
                stack.append(nxt)
    return seen


def audit_repo(repo: Path) -> list[Finding]:
    rows = parse_rows(repo)
    fmaps: dict[Path, dict[str, list[str]]] = {}
    out: list[Finding] = []
    for row in rows:
        d = Path(row.crate_dir)
        if d not in fmaps:
            fmaps[d] = feature_map(d)
        preds = leading_inner_cfgs(row.path)
        if preds is None:
            out.append(Finding(row, UNRESOLVED, why="source unreadable"))
            continue
        if not preds:
            out.append(Finding(row, NO_CFG))
            continue
        needs: set[str] = set()
        for p in preds:
            got = required_features(p)
            if got is None:
                out.append(Finding(row, UNRESOLVED, why=f"#![cfg({p})]"))
                break
            needs |= got
        else:
            if not needs:
                out.append(Finding(row, NO_FEATURE_REQ, why="platform/profile cfg only"))
                continue
            fmap = fmaps[d]
            bare = closure(fmap, row.features)
            resolved = closure(fmap, row.features + default_features(fmap))
            missing = needs - resolved
            if missing:
                out.append(Finding(row, EMPTY, needs=needs, missing=missing))
            elif needs - bare:
                out.append(Finding(row, EMPTY_NODEFAULT, needs=needs,
                                   missing=needs - bare,
                                   why="satisfied only via default features"))
            else:
                out.append(Finding(row, SATISFIED, needs=needs))
    return out


def selftest() -> None:
    """Pin the tokenizer and the closure in BOTH directions.

    Exits 2, never 1: an untrustworthy instrument is not the same finding as
    drift, and a narrowed tokenizer here reports a clean repo — which is the
    failure mode every ceiling in this family has already been bitten by.
    """
    fails: list[str] = []

    def rf(p: str) -> set[str] | None:
        return required_features(p)

    # ── the predicate reducer ──
    cases: list[tuple[str, set[str] | None, str]] = [
        ('feature = "a"', {"a"}, "bare feature"),
        ('all(feature = "a", feature = "b")', {"a", "b"}, "all() conjunction"),
        # any() is the shape cargo's AND-only required-features CANNOT express,
        # so it must not be reduced to a requirement.
        ('any(feature = "a", feature = "b")', None, "any() must not reduce"),
        ('not(feature = "a")', None, "not() INVERTS the implication"),
        ('all(feature = "a", not(feature = "b"))', None, "a not(feature) anywhere poisons it"),
        # Non-feature atoms are STRIPPED, not surrendered to — the repair that
        # took UNRESOLVED from 137 to a handful. set() != None here: the first
        # means "decidable, requires nothing", the second "cannot say".
        ('target_os = "macos"', set(), "platform-only: decidable, no requirement"),
        ("test", set(), "cfg(test): decidable, no requirement"),
        ('all(test, feature = "a")', {"a"}, "test conjunct stripped, feature kept"),
        ('all(feature = "a", not(target_os = "windows"))', {"a"},
         "not(platform) stripped, feature kept"),
        ('all(test, feature = "a", debug_assertions)', {"a"}, "two non-feature atoms stripped"),
        ('all(feature = "a", feature = "b", target_arch = "wasm32")', {"a", "b"},
         "two features survive a platform atom"),
        ('some_unmodelled_key = "x"', None, "an atom we do not model must not be guessed"),
        ('feature = "a-b_c.d"', {"a-b_c.d"}, "punctuated feature name"),
    ]
    for pred, want, name in cases:
        got = rf(pred)
        if got != want:
            fails.append(f"required_features({pred!r}) = {got}, want {want}  [{name}]")

    # ── conjunct splitting ──
    if split_conjuncts('all(a, not(b, c), d)') != ["a", "not(b, c)", "d"]:
        fails.append("split_conjuncts: split a comma INSIDE a nested predicate")
    if split_conjuncts('feature = "a"') != ['feature = "a"']:
        fails.append("split_conjuncts: a bare predicate is not its own conjunct")

    # ── defaults are part of the resolved set ──
    # This case exists because its absence produced 6 false EMPTY-AT-ROW
    # findings in one repo, caught by EXECUTING a target (3 passed, not 0).
    dfmap = {"default": ["authority"], "authority": [], "bevy_backend": []}
    if default_features(dfmap) != ["authority"]:
        fails.append("default_features: did not read the default list")
    if "authority" not in closure(dfmap, ["bevy_backend"] + default_features(dfmap)):
        fails.append("closure: a DEFAULT feature is missing from the resolved set")
    if "authority" in closure(dfmap, ["bevy_backend"]):
        fails.append("closure: seeded a default without being asked (bare set must stay bare)")

    # ── the closure ──
    fmap = {"big": ["small", "dep/x"], "small": [], "other": []}
    if closure(fmap, ["big"]) != {"big", "small"}:
        fails.append("closure: transitive same-package feature not followed")
    if "x" in closure(fmap, ["big"]):
        fails.append("closure: followed a dep/feat into our own namespace")
    if closure(fmap, ["missing_feat"]) != {"missing_feat"}:
        fails.append("closure: an undeclared name must still count as itself")
    # A cycle must terminate rather than hang; a hang is indistinguishable from
    # a slow repo and would take the whole report offline.
    if closure({"a": ["b"], "b": ["a"]}, ["a"]) != {"a", "b"}:
        fails.append("closure: cyclic features not handled")

    # ── the source scanner ──
    import tempfile

    def scan(body: str) -> list[str] | None:
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as fh:
            fh.write(body)
            name = fh.name
        try:
            return leading_inner_cfgs(name)
        finally:
            Path(name).unlink(missing_ok=True)

    if scan('#![cfg(feature = "a")]\n#![cfg(feature = "b")]\nfn main() {}\n') != [
        'feature = "a"',
        'feature = "b"',
    ]:
        fails.append("scanner: two ANDed inner cfgs not both collected")
    if scan('//! doc\n#![allow(dead_code)]\n\n#![cfg(feature = "a")]\nfn f() {}\n') != [
        'feature = "a"'
    ]:
        fails.append("scanner: doc comment or #![allow] terminated the scan early")
    # An OUTER #[cfg] gates ONE ITEM, not the file. Counting it would report a
    # target as empty when only one function is gated — a false EMPTY-AT-ROW,
    # which is the direction that wastes somebody's afternoon.
    if scan('#[cfg(feature = "a")]\nfn f() {}\n') != []:
        fails.append("scanner: an OUTER #[cfg] was read as gating the file")
    # ...and an inner cfg AFTER the first item cannot exist in real Rust, so
    # finding one means the scan ran past the item and is untrustworthy.
    if scan('fn f() {}\n#![cfg(feature = "a")]\n') != []:
        fails.append("scanner: scanned past the first item")
    if scan("fn main() {}\n") != []:
        fails.append("scanner: invented a cfg in a file with none")
    if leading_inner_cfgs("/nonexistent/definitely/not/here.rs") is not None:
        fails.append("scanner: an unreadable file reported as 'no cfg' rather than None")

    if fails:
        for f in fails:
            print(f"selftest FAIL: {f}", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repos", nargs="*", help="repo paths; default = all contract repos")
    ap.add_argument("--workspace", default=str(Path.home() / "git"))
    ap.add_argument("--show-unresolved", action="store_true")
    args = ap.parse_args(argv)

    selftest()

    repos = [Path(p).resolve() for p in args.repos] or derive_repos(Path(args.workspace))
    tally = {EMPTY: 0, EMPTY_NODEFAULT: 0, SATISFIED: 0, NO_FEATURE_REQ: 0,
             UNRESOLVED: 0, NO_CFG: 0}
    empties: list[Finding] = []
    unresolved: list[Finding] = []

    print(f"{'repo':<24} {'rows':>6} {'#![cfg]':>8} {'SATISFIED':>10} "
          f"{'no-feat-req':>12} {'via-default':>12} {'EMPTY':>6} {'UNRESOLVED':>11}")
    for repo in repos:
        found = audit_repo(repo)
        per = {k: sum(1 for f in found if f.verdict == k) for k in tally}
        for k in tally:
            tally[k] += per[k]
        empties += [f for f in found if f.verdict == EMPTY]
        unresolved += [f for f in found if f.verdict == UNRESOLVED]
        with_cfg = sum(per[k] for k in tally if k != NO_CFG)
        print(f"{repo.name:<24} {len(found):>6} {with_cfg:>8} {per[SATISFIED]:>10} "
              f"{per[NO_FEATURE_REQ]:>12} {per[EMPTY_NODEFAULT]:>12} {per[EMPTY]:>6} "
              f"{per[UNRESOLVED]:>11}")

    with_cfg = sum(tally[k] for k in tally if k != NO_CFG)
    print(f"\n{sum(tally.values())} rows over {len(repos)} repo(s); "
          f"{with_cfg} carry a leading #![cfg]; "
          f"{tally[EMPTY]} EMPTY-AT-ROW, {tally[UNRESOLVED]} UNRESOLVED "
          f"(not clean — needs a per-site read).")

    if empties:
        print("\nEMPTY-AT-ROW — the row builds, and compiles the target to NOTHING:")
        for f in empties:
            print(f"  {f.row.label}")
            print(f"      row     = {f.row.features}")
            print(f"      #![cfg] needs {sorted(f.needs)}  MISSING {sorted(f.missing)}")
            print(f"      {f.row.path}")
    if args.show_unresolved and unresolved:
        print("\nUNRESOLVED — a predicate this report declines to rule on:")
        for f in unresolved:
            print(f"  {f.row.label}  {f.why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
