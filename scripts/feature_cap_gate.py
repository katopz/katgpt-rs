#!/usr/bin/env python3
"""Feature-cap growth guard (Plan 635) — no publishable crate grows past the registry's cap.

crates.io refuses a crate with more than 300 features (katgpt-core 0.5.0 was refused at 414,
Issue 992; Plan 635 tranche 1 cut it to 278 and it published 2026-11-11). Nothing stopped the
count growing back: every new primitive ships behind a flag by rule, so the cap is reached by
ordinary work, and the refusal used to arrive only at upload time. This gate moves it to push
time.

What is counted is what the registry counts (Plan 635's census, matched against crates.io's own
refusal at 414): every key of the manifest's ``[features]`` table, ``default`` included, PLUS one
implicit feature per optional dependency that no feature row names as ``dep:<name>`` (cargo
synthesises those, and they ride the index like any other).

Population: every tracked ``Cargo.toml`` in this repo whose ``[package]`` does not say
``publish = false``. The crate that ships (katgpt-core) is a MUST-SEE member: a walk that does not
find it is BLIND, never green.

Verdicts, per crate: ``ok`` below the warn band · ``WARN`` within ``WARN_MARGIN`` of the cap (exit 0,
printed so the headroom is visible on every push) · ``OVER`` above the cap (exit 1, before the
registry refuses). A crate granted a raised limit gets a row in ``GRANTED`` with the grant's
reference — the cap is a pin, never a guess.

RATCHET (Plan 635 Tranche 2, verdict Q4): a crate pinned in ``scripts/feature_count_pins.txt``
must hold its count at ``ceiling + live grants`` exactly — GROWTH above it (a new primitive must
land under an existing family umbrella / parent row, or carry a ``grant`` row with a reason),
LOOSE below it (re-pin the ceiling down so freed slots never refill silently), STALE for a grant
naming a feature the manifest no longer has. All three exit 1. Pins apply to the repo walk only,
never to ``--manifest`` (a publish lane's rendered manifest is a different file).

Usage:
    python3 scripts/feature_cap_gate.py                    # the verdict, this repo
    python3 scripts/feature_cap_gate.py --manifest PATH    # one manifest (a publish lane's
                                                           # rendered Cargo.toml)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_safe  # noqa: E402

console_safe.apply()

ROOT = Path(__file__).resolve().parent.parent

# crates.io's default per-crate feature limit.
REGISTRY_CAP = 300
# Warn this many features before the cap (Plan 635: ~one tranche of normal flag growth).
WARN_MARGIN = 30
# Crates the registry granted a raised limit: name -> (cap, reference). Empty — none granted.
GRANTED: dict[str, tuple[int, str]] = {}
PINS = ROOT / "scripts" / "feature_count_pins.txt"
# The crate that ships; the walk must see it.
MUST_SEE = ("katgpt-core",)


def feature_count(manifest: dict) -> int:
    """The registry's count for one parsed manifest: the [features] keys plus the implicit
    feature of every optional dependency no row names as ``dep:<name>``."""
    feats = manifest.get("features", {}) or {}
    named_dep = {
        e[4:]
        for entries in feats.values()
        for e in entries
        if isinstance(e, str) and e.startswith("dep:")
    }
    tables = [manifest.get("dependencies", {}) or {}]
    for target in (manifest.get("target", {}) or {}).values():
        tables.append((target or {}).get("dependencies", {}) or {})
    optional = {
        name
        for table in tables
        for name, spec in table.items()
        if isinstance(spec, dict) and spec.get("optional") is True
    }
    implicit = {d for d in optional if d not in named_dep and d not in feats}
    return len(feats) + len(implicit)


def publishable(manifest: dict) -> str | None:
    """The package name when the manifest is a package that may publish, else None."""
    pkg = manifest.get("package")
    if not isinstance(pkg, dict):
        return None
    if pkg.get("publish") is False or pkg.get("publish") == []:
        return None
    name = pkg.get("name")
    return name if isinstance(name, str) else None


def verdict(name: str, count: int) -> tuple[str, int]:
    """(``ok`` | ``WARN`` | ``OVER``, the cap applied) for one crate."""
    cap = GRANTED.get(name, (REGISTRY_CAP, ""))[0]
    if count > cap:
        return "OVER", cap
    if count > cap - WARN_MARGIN:
        return "WARN", cap
    return "ok", cap


def parse_pins(text: str) -> tuple[dict[str, int], dict[str, dict[str, str]], list[str]]:
    """(ceilings, grants crate -> {feature: reason}, malformed lines) from the pin file text."""
    ceilings: dict[str, int] = {}
    grants: dict[str, dict[str, str]] = {}
    bad: list[str] = []
    for raw in text.splitlines():
        body, _, reason = raw.partition("#")
        parts = body.split()
        if not parts:
            continue
        if parts[0] == "ceiling" and len(parts) == 3 and parts[2].isdigit():
            if parts[1] in ceilings:
                bad.append(raw)
            ceilings[parts[1]] = int(parts[2])
        elif parts[0] == "grant" and len(parts) == 3 and reason.strip():
            grants.setdefault(parts[1], {})[parts[2]] = reason.strip()
        else:
            bad.append(raw)
    for crate in grants:
        if crate not in ceilings:
            bad.append(f"grant for {crate} with no ceiling row")
    return ceilings, grants, bad


def ratchet(name: str, count: int, names: set[str], ceiling: int, grants: dict[str, str]) -> list[str]:
    """Ratchet findings for one pinned crate (empty = held)."""
    findings = [
        f"STALE {name}: grant `{f}` names no [features] row — delete the grant"
        for f in sorted(grants)
        if f not in names
    ]
    allowed = ceiling + sum(1 for f in grants if f in names)
    if count > allowed:
        findings.append(
            f"GROWTH {name}: {count} > {allowed} (ceiling {ceiling} + {allowed - ceiling} grants)"
            " — land the new primitive under an existing family umbrella / parent row,"
            " or add a `grant` row with its reason to scripts/feature_count_pins.txt"
        )
    elif count < allowed:
        findings.append(
            f"LOOSE {name}: {count} < {allowed} — rows went; lower `ceiling {name}` to"
            f" {count - (allowed - ceiling)} in scripts/feature_count_pins.txt"
        )
    return findings


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def manifests() -> list[Path]:
    """Tracked package manifests: the root and every crate (vendor/target never tracked here)."""
    found = [ROOT / "Cargo.toml"] + sorted(ROOT.glob("crates/*/Cargo.toml"))
    return [p for p in found if p.is_file()]


def judge(
    paths: list[Path], must_see: tuple[str, ...], pins: str | None = None
) -> tuple[int, list[str], str]:
    """Exit code + report lines over ``paths`` + the least-headroom crate (``""`` if none).
    ``pins`` is the ratchet pin-file text (``None`` = no ratchet, the ``--manifest`` lane)."""
    ceilings, grants, bad = parse_pins(pins) if pins is not None else ({}, {}, [])
    lines: list[str] = [f"  MALFORMED pin: {b!r}" for b in bad]
    over = len(bad)
    seen: set[str] = set()
    tightest: tuple[int, str] = (sys.maxsize, "")
    for path in paths:
        manifest = load(path)
        name = publishable(manifest)
        if name is None:
            continue
        seen.add(name)
        count = feature_count(manifest)
        v, cap = verdict(name, count)
        over += v == "OVER"
        lines.append(f"  {v:<4} {name}: {count}/{cap} features ({cap - count} headroom)")
        if name in ceilings:
            names = set(manifest.get("features", {}) or {})
            findings = ratchet(name, count, names, ceilings[name], grants.get(name, {}))
            over += len(findings)
            lines.extend(f"  {f}" for f in findings)
            if not findings:
                lines.append(f"  held {name}: ratchet {count} = ceiling {ceilings[name]} + grants")
        if cap - count < tightest[0]:
            tightest = (cap - count, f"max {name} {count}/{cap}, {cap - count} headroom")
    blind = [m for m in must_see if m not in seen]
    blind += [f"{c} (pinned)" for c in ceilings if c not in seen]
    if blind:
        lines.append(f"  BLIND: the walk never saw {blind} — a cap verdict over nothing is not green")
        return 2, lines, tightest[1]
    return (1 if over else 0), lines, tightest[1]


def gate_selftest() -> list[str]:
    """Arms over the counting and the verdict arithmetic, run on every invocation."""
    failures: list[str] = []

    def manifest_text(n_rows: int, publish: str = "", extra: str = "") -> str:
        rows = "\n".join(f'f{i} = []' for i in range(n_rows - 1))
        return (
            f'[package]\nname = "probe"\nversion = "0.1.0"\n{publish}\n'
            f"{extra}\n[features]\ndefault = []\n{rows}\n"
        )

    with tempfile.TemporaryDirectory() as tmp:
        def count_of(text: str) -> int:
            p = Path(tmp) / "Cargo.toml"
            p.write_text(text, encoding="utf-8")
            return feature_count(load(p))

        if count_of(manifest_text(278)) != 278:
            failures.append("the default key must count (278 rows -> 278)")
        implicit = '[dependencies]\nx = { version = "1", optional = true }\n'
        if count_of(manifest_text(10, extra=implicit)) != 11:
            failures.append("an optional dep no row names as dep: must add one implicit feature")
        named = implicit + "\n"
        text = manifest_text(10, extra=named).replace("default = []", 'default = ["dep:x"]')
        if count_of(text) != 10:
            failures.append("an optional dep named as dep:x must add no implicit feature")
        p = Path(tmp) / "Cargo.toml"
        p.write_text(manifest_text(5, publish="publish = false"), encoding="utf-8")
        if publishable(load(p)) is not None:
            failures.append("publish = false must leave the population")
    c, g, b = parse_pins("ceiling k 10\ngrant k fam  # umbrella\ngrant k gone # x\nceiling k\n")
    if c != {"k": 10} or g != {"k": {"fam": "umbrella", "gone": "x"}} or b != ["ceiling k"]:
        failures.append(f"parse_pins: got {c} {g} {b}")
    if parse_pins("grant k f\n")[2] == []:
        failures.append("a grant with no reason must be MALFORMED")
    names = {f"f{i}" for i in range(11)} | {"fam"}
    arms = (
        (12, {"fam": "u"}, 11, []),  # 11 + 1 live grant = 12 held
        (13, {"fam": "u"}, 11, ["GROWTH"]),
        (11, {"fam": "u"}, 11, ["LOOSE"]),
        (12, {"fam": "u", "gone": "x"}, 11, ["STALE"]),
    )
    for count, gr, ceil, want in arms:
        got = [f.split()[0] for f in ratchet("k", count, names, ceil, gr)]
        if got != want:
            failures.append(f"ratchet(count={count}, grants={sorted(gr)}) = {got}, want {want}")
    if "lower `ceiling k` to 10" not in ratchet("k", 11, names, 11, {"fam": "u"})[0]:
        failures.append("LOOSE must print the re-pin value (count - live grants)")
    for count, want in ((269, "ok"), (270, "ok"), (271, "WARN"), (300, "WARN"), (301, "OVER")):
        got = verdict("probe", count)[0]
        if got != want:
            failures.append(f"verdict({count}) = {got}, want {want}")
    return failures


def main(argv: list[str]) -> int:
    failures = gate_selftest()
    if failures:
        for f in failures:
            print(f"✗ feature_cap_gate selftest: {f}")
        return 2
    if len(argv) == 2 and argv[0] == "--manifest":
        code, lines, top = judge([Path(argv[1])], must_see=())
        if not lines:
            print(f"✗ feature_cap_gate: {argv[1]} is not a publishable package manifest")
            return 2
    else:
        code, lines, top = judge(manifests(), MUST_SEE, PINS.read_text(encoding="utf-8"))
    for line in lines:
        print(line)
    mark = {0: "✓", 1: "✗", 2: "✗"}[code]
    word = {0: "PASS", 1: "FAIL — over the registry cap or off the ratchet pin", 2: "BLIND"}[code]
    tail = f"; {top}" if top else ""
    print(f"{mark} feature_cap_gate: {word} (cap {REGISTRY_CAP}, warn band {WARN_MARGIN}{tail})")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
