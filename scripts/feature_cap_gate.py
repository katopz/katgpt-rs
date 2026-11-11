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


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def manifests() -> list[Path]:
    """Tracked package manifests: the root and every crate (vendor/target never tracked here)."""
    found = [ROOT / "Cargo.toml"] + sorted(ROOT.glob("crates/*/Cargo.toml"))
    return [p for p in found if p.is_file()]


def judge(paths: list[Path], must_see: tuple[str, ...]) -> tuple[int, list[str]]:
    """Exit code + report lines over ``paths``."""
    lines: list[str] = []
    seen: set[str] = set()
    over = 0
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
    blind = [m for m in must_see if m not in seen]
    if blind:
        lines.append(f"  BLIND: the walk never saw {blind} — a cap verdict over nothing is not green")
        return 2, lines
    return (1 if over else 0), lines


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
        code, lines = judge([Path(argv[1])], must_see=())
        if not lines:
            print(f"✗ feature_cap_gate: {argv[1]} is not a publishable package manifest")
            return 2
    else:
        code, lines = judge(manifests(), MUST_SEE)
    for line in lines:
        print(line)
    mark = {0: "✓", 1: "✗", 2: "✗"}[code]
    word = {0: "PASS", 1: "FAIL — over the registry cap", 2: "BLIND"}[code]
    print(f"{mark} feature_cap_gate: {word} (cap {REGISTRY_CAP}, warn band {WARN_MARGIN})")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
