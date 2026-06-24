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

import sys
from pathlib import Path

import tomllib


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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tomls = [root / "Cargo.toml", root / "crates" / "katgpt-core" / "Cargo.toml"]

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

    # README claim check
    print("\n## README claim check")
    print('  current README claim: "140+ default-on, 320+ total flags"')
    print(f"  actual default-on   : {len(grand_default)}")
    print(f"  actual total flags  : {len(grand_total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
