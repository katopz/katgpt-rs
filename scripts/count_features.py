#!/usr/bin/env python3
"""Precise feature-flag count audit for katgpt-rs workspace.

Counts:
  - default-on features (entries in the `default` array, excluding `default` itself)
  - total feature flags (all keys under [features], excluding `default`)
  - net new opt-in flags (total - default)

Reads [features] from every Cargo.toml in the workspace (root + crates/*).
For workspace-level passthrough features (foo = ["katgpt-core/foo"]), counts
the workspace entry AND notes the underlying core feature.

Usage:
    python3 scripts/count_features.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from toml_compat import tomllib  # Issue 1245 T1: 3.11+ stdlib, else the tomli backport

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_doc_audit import iter_cargo_manifests


# Deliberately permissive: up to three intervening words, and "default
# features" as well as "default-on". Canaried at each widening — the first
# version required a literal "N flags" / "N total flags" / "N feature flags",
# so "999 tunable flags" and "999 default features" both sailed past a sweep
# whose whole job was catching phrasings nobody predicted. A sweep that only
# recognises the shapes already in `claims` is decoration.
SWEEP = (r"\b(\d+)\+?\s+(?:[A-Za-z][\w-]*\s+){0,3}flags?\b"
         r"|\b(\d+)\+?\s+(?:[A-Za-z][\w-]*\s+){0,3}default(?:-on|\s+features)\b")

# The RECOGNISED phrasings, with the field each capture group carries. One
# canonical pair (total, default-on) is asserted everywhere it appears; a doc
# wanting a different scope must add a named pattern here, which makes the new
# quantity reviewable instead of silently authoritative. Anything SWEEP finds
# and no pattern here covers is a hard failure, not a silent skip.
#
# Module-level since Issue 789: `selftest` asserts each pattern still matches
# the shape it was written for, and a second copy of the list would be a second
# thing to get wrong (Issue 755's rule, one file over).
CLAIMS = [
    (r"(\d+)\s+feature flags\s*\((\d+)\s+default-on", ("total", "default")),
    (r"\*\*(\d+)\s+GOAT-proved default-on features\*\*\s*\((\d+)\s+total flags\)",
     ("default", "total")),
    (r"\*\*(\d+)\s+feature flags\*\*\s+with\s+\*\*(\d+)\s+default-on",
     ("total", "default")),
    (r"feature flag table\s*\((\d+)\s+flags\)", ("total",)),
    (r"The full set\s*\((\d+)\s+flags\b", ("total",)),
]


def load_features(path: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Return (default_on, all_flags, raw_map) for a Cargo.toml [features] table."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    feats = data.get("features", {})
    if not feats:
        return set(), set(), {}
    all_flags = {k for k in feats if k != "default"}
    raw_default = feats.get("default", [])
    # default entries may be "katgpt-core/foo" (passthrough) or "foo"
    default_on = set()
    for entry in raw_default:
        # strip crate prefix for passthroughs
        name = entry.split("/", 1)[-1]
        default_on.add(name)
    default_on.discard("default")
    return default_on, all_flags, feats


def line_of(text: str, idx: int) -> int:
    """1-indexed line number of offset `idx` in `text`.

    ONE copy since Issue 790 T3. There were two identical closures — `line_of`
    inside the per-doc loop and `line_of_readme` below it — and
    `arm_reach_audit` reported the `+ 1` in BOTH surviving an off-by-one flip,
    because neither was reachable from any arm. The `+ 1` is the 0-to-1-indexed
    conversion, and getting it wrong sends whoever reads a finding to the wrong
    line: the "reported line is not the defect" hazard `markdown_fence_gate`'s
    docstring warns about, one file over.
    """
    return text.count("\n", 0, idx) + 1


def outside_parens(line: str) -> str:
    """`line` with every parenthesised span removed, nesting-aware.

    Only names OUTSIDE parentheses are entries in README's "Default features
    include:" list. Inside them is commentary that legitimately backticks
    non-default identifiers — "implies `engram`", "alias for
    `sense_composition`", function names like `poincare_navigate_into`.
    Counting those made the check demand that prose mentions be default-on,
    which is not the rule.

    Extracted from `main` by Issue 789 so the rule has one home and an arm can
    reach it; `main` is the only caller.
    """
    depth, outside = 0, []
    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            outside.append(ch)
    return "".join(outside)


def selftest() -> list[str]:
    """Known-answer arms over the three rules that decide this gate's verdict.

    Issue 789. The module comments say SWEEP was "canaried at each widening"
    and they are truthful about the history — but the canary was a person at a
    terminal, so nothing re-ran it and a later narrowing would be silent. These
    are the arms, and the inputs are the exact phrasings the comments record
    having escaped: a regression to any earlier version of SWEEP reds here.

    Pure string work plus one temp manifest — runs unconditionally in `main()`.
    """
    import tempfile

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    # ── SWEEP: an unrecognised claim-shaped number must NOT slip past ──────
    def swept(text: str) -> list[str]:
        return [m.group(0) for m in re.finditer(SWEEP, text)]

    # The two measured escapes the widening was made for. A sweep that only
    # recognises the shapes already in `claims` is decoration.
    eq("'999 tunable flags' is claim-shaped", swept("999 tunable flags"),
       ["999 tunable flags"])
    eq("'999 default features' is claim-shaped", swept("999 default features"),
       ["999 default features"])
    eq("the plain shape", swept("320 flags"), ["320 flags"])
    eq("the `+` suffix", swept("320+ total flags"), ["320+ total flags"])
    eq("singular 'flag'", swept("1 flag"), ["1 flag"])
    eq("'default-on' without 'features'", swept("140+ default-on"), ["140+ default-on"])
    eq("three intervening words is the documented bound",
       bool(swept("999 very oddly named flags")), True)
    eq("FOUR intervening words is past the bound (and so unswept)",
       swept("999 very oddly named extra flags"), [])
    eq("a number with no claim noun is not claim-shaped", swept("320 crates"), [])
    eq("a claim noun with no number is not claim-shaped",
       swept("the flags are documented"), [])

    # ── CLAIMS: each named pattern still matches the shape it was written for
    # Indexed by POSITION and floored by length: a reordering that silently
    # re-pairs a pattern with the wrong (total, default) field order is the
    # bug this arm exists for, and it is invisible to a set comparison.
    samples = [
        ("568 feature flags (197 default-on)", ("568", "197"), ("total", "default")),
        ("**197 GOAT-proved default-on features** (568 total flags)",
         ("197", "568"), ("default", "total")),
        ("**568 feature flags** with **197 default-on**",
         ("568", "197"), ("total", "default")),
        ("feature flag table (568 flags)", ("568",), ("total",)),
        ("The full set (568 flags)", ("568",), ("total",)),
    ]
    eq("every CLAIMS row has a sample arm", len(CLAIMS), len(samples))
    for (rx, fields), (sample, want_groups, want_fields) in zip(CLAIMS, samples):
        m = re.search(rx, sample)
        eq(f"CLAIMS pattern {rx[:34]!r}", m.groups() if m else None, want_groups)
        eq(f"CLAIMS fields  {rx[:34]!r}", fields, want_fields)
        # The pair must be READ in the pattern's declared order, which is what
        # `zip(fields, m.groups())` does in `main`. 568 and 197 are not
        # interchangeable, and two of the five rows declare them reversed.
        if m:
            eq(f"CLAIMS pairing {rx[:34]!r}",
               dict(zip(fields, m.groups())),
               {"total": "568", "default": "197"} if len(fields) == 2
               else {"total": "568"})

    # ── load_features: passthrough stripping and the `default` discard ─────
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "Cargo.toml"
        p.write_text(
            '[features]\n'
            # `katgpt-core/default` is the realistic trigger for the `default`
            # discard: the passthrough strip turns it into the literal name
            # "default", which is not a flag. Without an entry of this shape
            # the discard is unreachable and the arm reads INERT (measured).
            'default = ["alpha", "katgpt-core/beta", "katgpt-core/default"]\n'
            'alpha = []\n'
            'beta = []\n'
            'gamma = []\n', encoding="utf-8")
        default_on, all_flags, _raw = load_features(p)
        eq("a passthrough default entry is stripped to its feature name",
           sorted(default_on), ["alpha", "beta"])
        eq("`default` itself is not a flag", sorted(all_flags),
           ["alpha", "beta", "gamma"])
        eq("opt-in is the complement", sorted(all_flags - default_on), ["gamma"])

        p.write_text("[package]\nname = \"x\"\n", encoding="utf-8")
        eq("a manifest with no [features] table counts nothing",
           load_features(p), (set(), set(), {}))

    # ── outside_parens: commentary is not a list entry ────────────────────
    eq("a parenthesised name is not a list entry",
       sorted(re.findall(r"`([a-z0-9_]+)`",
                         outside_parens("`alpha`, `beta` (implies `engram`)"))),
       ["alpha", "beta"])
    eq("nesting is tracked",
       sorted(re.findall(r"`([a-z0-9_]+)`",
                         outside_parens("`alpha` (a (`nested`) `inner`) `beta`"))),
       ["alpha", "beta"])
    eq("an unbalanced closer does not go negative and swallow the tail",
       outside_parens("a) b"), "a b")

    # ── line_of: the 0-to-1-indexed conversion ────────────────────────────
    # ⚑ Issue 790 T3. This was two identical closures inside `main`, and the
    # `+ 1` in both survived an off-by-one flip because no arm could reach
    # either. A wrong line number sends the reader to the wrong line.
    eq("offset 0 is line 1", line_of("a\nb\nc", 0), 1)
    eq("an offset inside the first line is still line 1", line_of("ab\ncd", 1), 1)
    eq("the offset just past the first newline is line 2", line_of("ab\ncd", 3), 2)
    eq("the last line", line_of("a\nb\nc", 4), 3)
    eq("a single-line text is line 1", line_of("no newlines here", 5), 1)

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
    # The canary first. SWEEP is what stops an unrecognised phrasing from
    # passing silently, so a narrowed SWEEP does not produce a finding — it
    # produces a green over claims nothing read (exit 2, not 1).
    arm_failures = selftest()
    if arm_failures:
        print("✗ INSTRUMENT: count_features' own selftest does not pass, so the "
              "claim check below would be unreadable:")
        for f in arm_failures:
            print(f)
        return 2
    root = Path(__file__).resolve().parent.parent
    # EVERY workspace manifest, which is what the module docstring has always
    # claimed. It used to hardcode two (root + katgpt-core) and so measured a
    # SUBSET while reporting it as the workspace: 537/189 against an actual
    # 568/197. The gap is small because most crate features are passthrough
    # duplicates of root/core names, which is exactly why the under-scope
    # survived — it looked right.
    #
    # Discovered by walking, not by a `crates/*/Cargo.toml` glob: the glob
    # assumes every crate sits exactly one level down, so a nested crate that
    # defined features would be silently uncounted — the same class of bug one
    # level deeper. `iter_cargo_manifests` is shared with the doc auditors, so
    # all four checks agree on what "the workspace" means.
    tomls = [p for p in sorted(iter_cargo_manifests(root)) if load_features(p)[1]]

    print("=" * 72)
    print("katgpt-rs feature-flag audit")
    print("=" * 72)

    grand_default: set[str] = set()
    grand_total: set[str] = set()

    for toml in tomls:
        rel = toml.relative_to(root)
        default_on, all_flags, feats = load_features(toml)
        opt_in = all_flags - default_on
        print(f"\n## {rel}")
        print(f"  default-on : {len(default_on)}")
        print(f"  total flags: {len(all_flags)}")
        print(f"  opt-in     : {len(opt_in)}")
        if feats.get("default"):
            print(f"  default[] length: {len(feats['default'])}")
        grand_default |= default_on
        grand_total |= all_flags

    # Union view (dedup across root + core)
    print("\n" + "=" * 72)
    print("WORKSPACE UNION (deduped across root + katgpt-core)")
    print("=" * 72)
    print(f"  default-on (unique) : {len(grand_default)}")
    print(f"  total flags (unique): {len(grand_total)}")
    print(f"  opt-in (unique)     : {len(grand_total - grand_default)}")

    # ── README claim check ──────────────────────────────────────────────────
    # Two generations of blind spot, both green:
    #
    # 1. It printed a HARDCODED claim string ("140+ default-on, 320+ total
    #    flags") beside the measured numbers and left the comparison to a human.
    #    A check comparing a measurement against a constant it also owns is not
    #    a check; by 2026-09-01 the literal matched neither the README nor
    #    reality.
    # 2. The replacement parsed the README but used re.search — the FIRST match
    #    only. README states its counts in FOUR places in THREE phrasings, so
    #    the headline was validated and three sites (378/152, 373/155, 373) were
    #    structurally unreachable. Fixing one number and asserting it pinned the
    #    other three in place.
    #
    # So: every known phrasing is checked at every occurrence, AND any
    # claim-shaped number that no pattern recognises is a hard failure rather
    # than a silent skip. An unknown phrasing must not be able to pass.
    print("\n## doc claim check")
    actual_total, actual_default = len(grand_total), len(grand_default)
    expect = {"total": actual_total, "default": actual_default}

    # Every doc that states a flag count, not just README. examples/README.md
    # claimed "292 flags" — matching no manifest, current or historical — and
    # was invisible to a README-only check. One canonical pair (total,
    # default-on) is asserted everywhere it appears; a doc wanting a different
    # scope must add a named pattern here, which makes the new quantity
    # reviewable instead of silently authoritative.
    docs = ["README.md", "examples/README.md"]

    claims = CLAIMS

    failures: list[str] = []
    total_sites = 0

    for rel in docs:
        path = root / rel
        if not path.exists():
            failures.append(f"{rel}: listed in `docs` but missing — its claims "
                            f"are unchecked, which is how coverage silently shrinks")
            continue
        text = path.read_text(encoding="utf-8")

        covered: list[tuple[int, int]] = []
        sites = 0
        for rx, fields in claims:
            for m in re.finditer(rx, text):
                sites += 1
                covered.append(m.span())
                ln = line_of(text, m.start())
                for field, raw in zip(fields, m.groups()):
                    got = int(raw)
                    mark = "✓" if got == expect[field] else "✗"
                    print(f"  {mark} {rel}:{ln}: {field} = {got} "
                          f"(measured {expect[field]})")
                    if got != expect[field]:
                        failures.append(
                            f"{rel}:{ln} claims {field}={got}, "
                            f"measured {expect[field]}")

        if sites == 0:
            failures.append(
                f"{rel}: no parseable feature-flag claim found — either the doc "
                f"stopped stating one (drop it from `docs`) or the phrasing "
                f"changed and is now unchecked")
        total_sites += sites

        for m in re.finditer(SWEEP, text):
            if any(a <= m.start() < b for a, b in covered):
                continue
            failures.append(
                f"{rel}:{line_of(text, m.start())} unrecognised flag-count phrasing "
                f"{m.group(0)!r} — add it to `claims` or reword it; NOT checked")

    print(f"  … {total_sites} claim site(s) recognised across {len(docs)} doc(s)")

    # ── "Default features include: …" list contents ─────────────────────────
    # Two independent things go stale here. The names: README listed nine
    # OPT-IN flags (bandit, ppot, bt_rank, elf_sde, cna_steering, dash_attn,
    # bfcf_lfu_shard, rcd_residual, slod) under "Default features include:",
    # none of which appears in ANY default array — the stale-flag-state class,
    # in README rather than a plan header. And the tail: "and N more" was 58,
    # internally consistent with that line's own stale 155 and with nothing
    # else. N is defined here as "default-on flags not named on this line", so
    # it is computable and therefore assertable.
    readme_text = (root / "README.md").read_text(encoding="utf-8")


    dm = re.search(r"Default features include:(.*)", readme_text)
    if not dm:
        failures.append("README.md: 'Default features include:' block not found "
                        "— the list check silently covers nothing")
    else:
        line, ln = dm.group(1), line_of(readme_text, dm.start())
        # Parenthesised commentary is not a list entry — rule and rationale at
        # `outside_parens`, which Issue 789 extracted so an arm can reach it.
        toks = set(re.findall(r"`([a-z0-9_]+)`", outside_parens(line)))
        listed_default = toks & grand_default
        listed_optin = (toks & grand_total) - grand_default
        if listed_optin:
            failures.append(
                f"README.md:{ln} lists {len(listed_optin)} OPT-IN flag(s) as "
                f"default-on: {', '.join(sorted(listed_optin))}")
        else:
            print(f"  ✓ L{ln}: all {len(listed_default)} listed flags are default-on")
        nm = re.search(r"and (\d+) more", line)
        want_more = actual_default - len(listed_default)
        if not nm:
            failures.append(f"README.md:{ln} has no 'and N more' tail to check")
        elif int(nm.group(1)) != want_more:
            failures.append(
                f"README.md:{ln} says 'and {nm.group(1)} more'; "
                f"{len(listed_default)} named + measured {actual_default} default-on "
                f"=> 'and {want_more} more'")
        else:
            print(f"  ✓ L{ln}: 'and {want_more} more' closes the default-on count")

    if failures:
        print("  ✗ doc claims have drifted from the manifests")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(f"  ✓ all claim sites across {len(docs)} doc(s) match the manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
