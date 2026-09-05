#!/usr/bin/env python3
"""Find cargo targets whose whole-file `#![cfg(...)]` can zero them SILENTLY.

A test file that opens with `#![cfg(feature = "x")]` compiles to an empty
binary when `x` is off. Cargo then prints

    running 0 tests
    test result: ok. 0 passed; 0 failed; ...

and exits 0. That is indistinguishable from a real pass, and it is how eleven
real assertions in riir-train's `nora_phase1_hooks` reported as a green suite
having run none (riir-train `5821cba9`), and how riir-clippy's `t063` — the
harness behind a whole benchmark — did the same (riir-clippy `19beece`).

`required-features` is the fix, and it is not cosmetic: it changes the outcome
from a silent green to

    error: target `t063_tpr_structure` in package `riir-clippy` requires the
    features: `tpr_structure`                                   (exit 101)

**The `#![cfg]` protects the COUNT. `required-features` protects the READER.**
Both are needed; neither substitutes for the other.

## What this is NOT

A **report, not a gate** — always exit 0, same discipline as
`ci_gate_coverage.py` and `staged_set_audit.py`. Two reasons it must not block:

1. A `cfg` on `target_os` / `target_arch` / `miri` **cannot** be expressed as
   `required-features`. Those files are correctly gated and are reported in a
   separate class, never as a defect.
2. `any(...)` of several features is a legitimate shape that cargo's
   AND-only `required-features` cannot express (riir-ai's `ci_feature_guard.sh`
   documents one). Reported as ITS own class rather than flattened into the
   defect list — a report that cries wolf on the one shape cargo cannot fix
   gets ignored on the shapes it can.

## Population is derived, expectations are committed

The repo set comes from the workspace walk (a root `BOUNDARY.md` **and** a
`.git` dir), never a typed list — deriving both the population and the
expectation from the same walk is what makes a cross-repo gate permanently
green.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Target dirs and the cargo table that declares them.
TARGET_KINDS = {"tests": "test", "benches": "bench", "examples": "example"}

# `cfg` predicates that `required-features` cannot express. A file gated on one
# of these is correctly gated; it is not a finding.
NON_FEATURE_PREDICATES = (
    "target_os",
    "target_arch",
    "target_family",
    "target_env",
    "target_pointer_width",
    "target_endian",
    "miri",
    "debug_assertions",
    "unix",
    "windows",
    "test",
)

# A whole-file inner attribute at the top of a file. `#![cfg(...)]` only —
# `#![allow]`, `#![doc]` etc. are not gates.
INNER_CFG = re.compile(r"^\s*#!\s*\[\s*cfg\s*\(", re.MULTILINE)

FEATURE_IN_CFG = re.compile(r'feature\s*=\s*"([^"]+)"')

# The PROFILE dimension, split by DIRECTION. Pooling the two would repeat the
# exact error this file documents for platform gates ("the negated and positive
# cases differ by the single token `not(`, and are OPPOSITE in severity"):
#
#   not(debug_assertions)  -> RELEASE-only. Silent under plain `cargo test`,
#                             i.e. the default invocation. The severe one.
#   debug_assertions       -> DEBUG-only. Runs by default and is silent under
#                             `--release` — the profile the perf rule
#                             (`.docs/10_audits/debug_release_profile_axis.md`) tells everyone to
#                             run gates in. Less severe, not harmless.
#
# Matched on the parenthesised form so `not(all(debug_assertions, ...))` and a
# bare `debug_assertions` elsewhere in the same cfg cannot be confused.
NOT_DEBUG_ASSERTIONS = re.compile(r"not\s*\(\s*debug_assertions\s*\)")

# A target whose filename says its green IS the evidence for a promotion or a
# claim. This vocabulary is COMMITTED here rather than derived from the corpus:
# deriving "which names look load-bearing" from the files present is the
# vocabulary-vs-population trap (Issue 703) — a repo that renames its gates
# would shrink the class into a confident zero.
#
# Matched on TOKENS, never as substrings. A substring match on "gate" claims
# `aggregate`, `delegate`, `propagate`, `mitigate` and `investigate`; on "g<N>"
# it claims nothing useful at all. Both directions are pinned in selftest().
LOAD_BEARING_TOKENS = frozenset(
    {
        "goat",
        "gate",
        "gates",
        "drill",
        "invariant",
        "invariants",
        "guard",
        "pin",
        "proof",
        "conservation",
        "safety",
        "security",
        "audit",
        # ── added 2026-09-03, after Plan 580 T5.3 armed two targets that this
        # set did not classify: `certified_frontier_correctness` (31 assertions)
        # and `bench_688_certified_frontier_alloc_check` (an alloc budget).
        # Both are gates by any reading, both were SILENT-NOW, and neither was
        # in T2's armed 39 *because the classifier could not see them*. So the
        # `max_load_bearing = 0` pin was a green over a population that excluded
        # a whole naming dialect.
        #
        # Each of these was measured across all 2,157 workspace test/bench
        # targets before being added (see the token table in `.docs/10_audits/cfg_gated_silent_zero_pass.md`),
        # and every one names a property the file exists to FAIL on:
        "alloc",         # G4 in this repo's GOAT convention: `*_alloc_check`
        "correctness",   # G1
        "determinism",   # the bit-identity arm of nearly every gate here
        "equivalence",
        "soundness",
        "floor",         # the Report-the-Floor rule (AGENTS.md), `conformal_floor_*`
        "grad",          # `*_backward_grad_check` — numerical-gradient gates
        # DELIBERATELY NOT added, and the reason is the same one that keeps
        # "gate" a token rather than a substring: `budget` admits a sweep
        # (`bench_578_mcts_budget_sweep`) and a config (`game_budgets`);
        # `check` admits any smoke test; `calibration` names a measurement
        # record, not an assertion; `coverage`/`regression`/`bound` matched
        # nothing new at all. A column that cries wolf stops being read.
        #
        # An explicit compound, not a substring rule. riir-clippy's
        # `t40_fixer_regate_harness` is a re-gate harness, and the only way a
        # token matcher sees it is by naming it — a substring rule for "gate"
        # would re-admit aggregate/delegate/propagate/mitigate/investigate,
        # which is the false-positive class that makes the column unreadable.
        # Trading one named compound for five false positives is the right way
        # round; add compounds here as they are found.
        "regate",
        # ── added 2026-09-06, by the FIRST corpus token table run over all 16
        # repos rather than over katgpt-rs alone. The set above was measured
        # against 2,157 katgpt-rs-era target names; the workspace corpus is
        # 3,081, and the sibling dialects were invisible in it. Before this,
        # `silent_now_load_bearing` was **0 in all 16 repos** — a zero that
        # read as "nobody ships a silent load-bearing gate" and actually meant
        # "the classifier speaks one repo's dialect". Widening it takes the
        # workspace count 0 -> 12, TWO OF THEM IN katgpt-rs ITSELF, whose
        # `max_load_bearing = 0` pin had been green over a population that
        # excluded them.
        #
        # Each was checked against every one of its corpus matches, not just
        # its SILENT-NOW ones, and each names a property the file exists to
        # FAIL on:
        "parity",         # 22 matches, ALL A-vs-B equivalence (GPU vs CPU,
                          # cudarc vs CPU, tokenizer, install-copy). Zero false
                          # positives; the same family as `equivalence`.
        "monotonicity",   # 4, all riir-chain slashing / domain-registry
                          # invariants. The same family as `invariant`.
        "roundtrip",      # 5, all encode/decode identity assertions.
        "integrity",      # 6, all genuine integrity assertions.
        "exactness",      # 2, riir-chain `ledger_exactness` — exact arithmetic.
        "reachability",   # 3. `taming_reachability` earns it: a reachability
                          # test's green IS the evidence that an authored goal
                          # can ever be won.
        # DELIBERATELY NOT added, measured the same way:
        # `e2e` (83, the largest class) and `persistence` name a SCOPE or a
        # subsystem, not a property to fail on — the same reason `check` was
        # rejected. `cost`/`throughput`/`overhead`/`latency`/`scale`/`growth`
        # name MEASUREMENTS, which is exactly why `calibration` was rejected.
        # `oracle` (3) names a technique, and in riir-chain vocabulary also a
        # component. `identity` (6) is a genuine homonym: 4 of its 6 are
        # `signer_identity` / `identity_matcher`, a domain NOUN.
        # `liveness`/`conformance`/`idempotence` matched nothing at all.
    }
)

# Explicit COMPOUNDS, checked against ADJACENT token pairs. Same principle as
# `regate` above — name the compound rather than loosening a token to a
# substring — but here neither half can be admitted alone, which is the whole
# argument for the mechanism:
#
#   `spec`  — 49 matches, and in THIS repo `spec_` means SPECULATIVE decoding
#             (`spec_reconciliation_bench`, and a `_demo`). A homonym.
#   `match` — 49 matches, of which `attn_match_*` (attention matching, a
#             subsystem) and `quest_match_tui` (a game match) are not gates.
#
# The compound admits neither, and picks up riir-chain's 39-target `*_spec_match`
# conformance convention — which katgpt-rs shares. Five of this repo's seven
# `*_spec_match` targets were only ever visible because they ALSO carry `g1`;
# `bridge_spec_match` and `pencil_spec_match` do not, and were invisible.
LOAD_BEARING_BIGRAMS = frozenset({"spec_match"})

# `g1`..`g<N>`, optionally with a variant suffix — the GOAT sub-gate naming
# convention (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free) as it
# is actually written across the workspace: `g16f`, `g2p`, `g2s`, `g9gov` are
# all real target names. A bare `g` is not one; the leading digit is required.
GATE_ORDINAL = re.compile(r"^g\d+[a-z0-9]*$")

# Token separators used across the workspace's target filenames: `_`, `-`, and
# `.` (the `bench_256_kv_outer.goat.rs` dialect, which is why `.` is here).
TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def is_load_bearing(*names: str) -> bool:
    """Does any name carry a load-bearing TOKEN? Substring matches excluded."""
    for name in names:
        toks = TOKEN_SPLIT.split(name.lower())
        for i, tok in enumerate(toks):
            # Depluralise rather than listing every plural: `gates`, `drills`,
            # `guards`, `proofs`, `audits` all appear, and a hand-listed set
            # misses whichever one is coined next. `drills` was a real miss.
            stem = tok[:-1] if tok.endswith("s") and len(tok) > 2 else tok
            if tok in LOAD_BEARING_TOKENS or stem in LOAD_BEARING_TOKENS:
                return True
            if GATE_ORDINAL.match(tok):
                return True
            # Adjacent-pair compounds, for the case where neither half can be
            # admitted alone without dragging in a homonym class.
            if i + 1 < len(toks) and f"{tok}_{toks[i + 1]}" in LOAD_BEARING_BIGRAMS:
                return True
    return False


@dataclass
class Finding:
    repo: str
    manifest: str
    kind: str
    name: str
    path: str
    features: list[str]
    predicates: list[str]
    declared: bool  # a [[test]]/[[bench]]/[[example]] row exists at all
    reason: str
    # Is EVERY gating feature reachable from this crate's `default`? If so the
    # target still runs on a plain `cargo test` and only vanishes under
    # `--no-default-features` — a real hazard, but a rarer one. If ANY gating
    # feature is default-off, a plain `cargo test` compiles the file to nothing
    # and reports a green 0-pass. That is the severity split, and without it
    # the headline count pools two populations an order of magnitude apart.
    default_on: bool = False
    # Does the filename say this target's green is evidence? Reported apart
    # because the severity is not the same: a silent zero on `scratch_probe`
    # costs a reader's time, and a silent zero on `plan414_..._goat` is a
    # promotion decision made over no measurement.
    load_bearing: bool = False
    # PROFILE axis (riir-ai `.issues/855` Class 2, 2026-09-03). A
    # `debug_assertions` term in the cfg is NOT another bucket — it is a second
    # DIMENSION, and it is deliberately allowed to overlap every class above,
    # `covered` included.
    #
    # Why it needs its own flag rather than sitting in `predicates`: every
    # other non-feature predicate is silent only in a configuration somebody
    # chose. `target_os` needs the wrong platform; `miri` needs miri;
    # `--no-default-features` needs the flag typed. `not(debug_assertions)`
    # is silent under `cargo test` — the DEFAULT invocation, on the right
    # machine, with no flags. It is the one gate whose green zero is what
    # everybody gets by default.
    #
    # And it survives the fix: adding a `required-features` row moves a target
    # into `covered`, which reads as "protected". A profile-gated target is
    # still a green zero in dev after that row lands. Measured: riir-gpu's
    # three `bench_734_*`/`bench_606_*` targets reported 0 errors in dev and
    # 6/8/4 in release, and a `compile_error!` planted in one of them did not
    # fire in dev at all.
    profile_gated: bool = False
    # Direction WITHIN the profile dimension. `not(debug_assertions)` is silent
    # under the DEFAULT invocation; a bare `debug_assertions` is silent under
    # `--release`, which is the profile the perf rule mandates for gates. One
    # number over both says nothing, same as for platform gates.
    release_only: bool = False


@dataclass
class RepoReport:
    repo: str
    scanned: int = 0
    gated: int = 0
    findings: list[Finding] = field(default_factory=list)
    platform_only: list[Finding] = field(default_factory=list)
    platform_except: list[Finding] = field(default_factory=list)
    cfg_test: list[Finding] = field(default_factory=list)
    any_of: list[Finding] = field(default_factory=list)
    # OVERLAPS every list above and `covered` — deliberately NOT part of the
    # partition, so the partition assertion is unaffected.
    profile: list[Finding] = field(default_factory=list)
    covered: int = 0

    def silent_now(self) -> list[Finding]:
        """Findings that zero on a PLAIN `cargo test` — the severe class."""
        return [f for f in self.findings if not f.default_on]

    def silent_latent(self) -> list[Finding]:
        """Findings that zero only under `--no-default-features`."""
        return [f for f in self.findings if f.default_on]

    @property
    def unexpressible(self) -> list[Finding]:
        """The three non-feature classes pooled — kept so the partition
        assertion (1,016 + 382 + 320 + 21 + 1 = 1,740) still holds and so the
        `--json` contract keeps its shape for existing consumers."""
        return self.platform_only + self.platform_except + self.cfg_test

    def silent_now_load_bearing(self) -> list[Finding]:
        """The severe class, restricted to targets whose green is evidence."""
        return [f for f in self.silent_now() if f.load_bearing]


def cfg_body(text: str) -> str | None:
    """The balanced body of the FIRST whole-file `#![cfg(...)]`, or None.

    Balanced-paren scan rather than a regex: `cfg(all(feature = "a",
    feature = "b"))` is the common shape and a non-greedy `\\)` stops at the
    first inner paren, silently reporting one feature where there are two.
    """
    m = INNER_CFG.search(text)
    if not m:
        return None
    i = text.index("(", m.start())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def default_closure(features: dict) -> set[str]:
    """Own-crate features reachable from `default`.

    Within-manifest only, and deliberately so: `dep/feat` and `dep:dep` entries
    enable a DEPENDENCY's feature, which cannot gate a `#![cfg(feature = ...)]`
    in THIS crate. Including them would over-approximate the default set and
    silently downgrade real findings to the latent class.
    """
    out: set[str] = set()
    stack = list(features.get("default", []) or [])
    while stack:
        f = stack.pop()
        if f in out or "/" in f or f.startswith("dep:") or f.startswith("?"):
            continue
        out.add(f)
        stack.extend(features.get(f, []) or [])
    return out


def scan_manifest(repo: Path, manifest: Path, rep: RepoReport) -> None:
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (tomllib.TOMLDecodeError, OSError):
        return

    # Declared targets, keyed by the FILE they build, not by their name.
    #
    # Keying on `(kind, name)` and matching it against the filename stem is
    # wrong, and wrong in the direction that manufactures findings: a row may
    # carry an explicit `path`, and then its `name` need not resemble the
    # file's stem at all. katgpt-rs's four `*.goat.rs` targets are declared as
    # `bench_256_kv_outer_goat` (underscored) pointing at
    # `bench_256_kv_outer.goat.rs` (dotted) with `required-features` already
    # present — a stem match reports all four as defects, and "fixing" them
    # adds a SECOND target for the same file, which cargo warns about and which
    # breaks `--test <name>` resolution.
    #
    # So resolve every row to an absolute path (explicit `path`, else the
    # conventional `<dir>/<name>.rs`) and key on that. `name` is kept only for
    # the suggested fix line.
    declared: dict[Path, bool] = {}
    declared_name: dict[Path, str] = {}
    crate_dir = manifest.parent
    for kind, dirname in (("test", "tests"), ("bench", "benches"), ("example", "examples")):
        for row in data.get(kind, []) or []:
            if not isinstance(row, dict):
                continue
            rel = row.get("path") or (
                f"{dirname}/{row['name']}.rs" if "name" in row else None
            )
            if rel is None:
                continue
            resolved = (crate_dir / rel).resolve()
            # A file may legitimately back two rows (different feature sets).
            # Covered if ANY row declares required-features — the question is
            # whether a reader can be silently fooled, and one guarded row is
            # enough to make the omission visible.
            declared[resolved] = declared.get(resolved, False) or bool(
                row.get("required-features")
            )
            declared_name.setdefault(resolved, row.get("name", ""))

    defaults = default_closure(data.get("features", {}) or {})
    crate = manifest.parent
    for dirname, kind in TARGET_KINDS.items():
        d = crate / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.rs")):
            rep.scanned += 1
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = cfg_body(text)
            if body is None:
                continue
            rep.gated += 1

            key = f.resolve()
            name = declared_name.get(key) or f.stem
            feats = sorted(set(FEATURE_IN_CFG.findall(body)))
            preds = sorted(
                {p for p in NON_FEATURE_PREDICATES if re.search(rf"\b{p}\b", body)}
            )
            has_rf = declared.get(key)

            base = Finding(
                repo=rep.repo,
                manifest=str(manifest.relative_to(repo)),
                kind=kind,
                name=name,
                path=str(f.relative_to(repo)),
                features=feats,
                predicates=preds,
                declared=key in declared,
                reason="",
                load_bearing=is_load_bearing(f.name, name),
            )

            base.profile_gated = "debug_assertions" in preds
            base.release_only = bool(NOT_DEBUG_ASSERTIONS.search(body))
            if base.profile_gated:
                # Recorded BEFORE the `has_rf` continue, on purpose: a covered
                # target with a profile gate is exactly the case this axis
                # exists to surface, and skipping it here would make the
                # column read zero the moment somebody does the easy fix.
                rep.profile.append(base)

            if has_rf:
                rep.covered += 1
                continue
            if not feats:
                # `required-features` cannot express any of these — but they are
                # three DIFFERENT things, and pooling them under the label
                # "platform" hid that for one measurement (Issue 713 T5):
                #
                #   not(target_arch = "wasm32")  compiles EVERYWHERE except one
                #                                arch. The inverse of a coverage
                #                                hazard. 11 of the 21.
                #   target_arch = "wasm32"       compiles ONLY there, so it runs
                #                                on no CI platform unless one
                #                                targets it. 2 of the 21 — the
                #                                only real coverage question.
                #   test                         a NO-OP in an integration
                #                                target: cargo passes --test, so
                #                                cfg(test) always holds. 8 of 21.
                #
                # The negated and positive cases differ by the single token
                # `not(`, and are OPPOSITE in severity. One number over both
                # says nothing.
                compact = body.replace(" ", "").replace("\n", "")
                if preds == ["test"]:
                    base.reason = "cfg(test) — a no-op in an integration target"
                    rep.cfg_test.append(base)
                elif "not(" in compact:
                    base.reason = "negated platform gate — compiles everywhere but one"
                    rep.platform_except.append(base)
                else:
                    base.reason = "positive platform gate — compiles ONLY there"
                    rep.platform_only.append(base)
                continue
            if "any(" in body.replace(" ", "") and len(feats) > 1:
                base.reason = "any(feature,...) — cargo's required-features is AND-only"
                rep.any_of.append(base)
                continue
            base.default_on = bool(feats) and all(x in defaults for x in feats)
            base.reason = (
                "declared without required-features"
                if base.declared
                else "auto-discovered, no [[%s]] row at all" % kind
            )
            rep.findings.append(base)


def manifests(repo: Path) -> list[Path]:
    out = []
    root = repo / "Cargo.toml"
    if root.is_file():
        out.append(root)
    for d in ("crates", "."):
        base = repo / d
        if not base.is_dir():
            continue
        for m in sorted(base.glob("*/Cargo.toml")):
            if m not in out and ".git" not in m.parts and "target" not in m.parts:
                out.append(m)
    return out


def audit(repo: Path) -> RepoReport:
    rep = RepoReport(repo=repo.name)
    for m in manifests(repo):
        scan_manifest(repo, m, rep)
    return rep


def derive_repos(workspace: Path) -> list[Path]:
    """A root BOUNDARY.md AND a `.git` DIR — never a typed list."""
    return sorted(
        d
        for d in workspace.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )


def selftest() -> None:
    """Pin the parse shapes. Runs on EVERY invocation.

    Without this the audit degrades silently: a regex regression makes it
    recognise fewer gates and still print a confident `0 findings`, which is
    the exact failure mode it exists to catch, committed by the tool that
    catches it.
    """
    cases = [
        # (source, expected cfg body or None)
        ('#![cfg(feature = "x")]\nfn a() {}', 'feature = "x"'),
        # Balanced-paren: the nested-all case a non-greedy regex truncates.
        (
            '#![cfg(all(feature = "a", feature = "b"))]\n',
            'all(feature = "a", feature = "b")',
        ),
        ("#![allow(clippy::pedantic)]\nfn a() {}", None),
        ('//! doc\n\n#![cfg(feature = "y")]\n', 'feature = "y"'),
        ("fn a() {}\n", None),
    ]
    for src, want in cases:
        got = cfg_body(src)
        assert got == want, f"cfg_body({src!r}) = {got!r}, want {want!r}"

    # The nested case must yield BOTH features — the bug a truncating scan hides.
    body = cfg_body('#![cfg(all(feature = "a", feature = "b"))]\n')
    assert sorted(set(FEATURE_IN_CFG.findall(body))) == ["a", "b"], "nested features lost"

    # A platform gate must carry no feature, so it lands in `unexpressible`
    # rather than the defect list.
    body = cfg_body('#![cfg(target_os = "macos")]\n')
    assert FEATURE_IN_CFG.findall(body) == [], "platform gate read as a feature gate"
    assert any(p in body for p in NON_FEATURE_PREDICATES), "platform predicate not recognised"

    # A mixed gate IS a finding: the feature half is expressible.
    body = cfg_body('#![cfg(all(target_os = "macos", feature = "gpu"))]\n')
    assert FEATURE_IN_CFG.findall(body) == ["gpu"], "mixed gate lost its feature"

    # The PROFILE dimension (riir-ai `.issues/855` Class 2). Pinned in both
    # directions because both failure modes are silent: a matcher that stops
    # seeing `debug_assertions` takes the column to a confident zero, and one
    # that fires on any file mentioning the word makes it unreadable.
    prof = cfg_body(
        '#![cfg(all(feature = "a", feature = "b", not(debug_assertions)))]\n'
    )
    prof_preds = {p for p in NON_FEATURE_PREDICATES if re.search(rf"\b{p}\b", prof)}
    assert "debug_assertions" in prof_preds, "profile term not recognised"
    # DIRECTION, pinned separately. Pooling the two halves is the error this
    # file already documents for platform gates, one axis over.
    assert NOT_DEBUG_ASSERTIONS.search(prof), "release-only direction not detected"
    assert not NOT_DEBUG_ASSERTIONS.search(
        cfg_body('#![cfg(all(feature = "a", debug_assertions))]\n')
    ), "a bare debug_assertions read as release-only — opposite severity"
    # A `not(...)` around something ELSE must not make the file release-only.
    assert not NOT_DEBUG_ASSERTIONS.search(
        cfg_body('#![cfg(all(not(target_os = "macos"), debug_assertions))]\n')
    ), "not(target_os) claimed by the profile-direction matcher"
    # ...and it must NOT swallow the feature half: this shape is a `findings`
    # entry (expressible) AND profile-gated. If the feature list were lost the
    # target would silently move to the platform class and never be reported
    # as needing a `required-features` row.
    assert sorted(set(FEATURE_IN_CFG.findall(prof))) == ["a", "b"], (
        "profile gate swallowed its features"
    )
    for clean in (
        '#![cfg(all(feature = "a", target_os = "macos"))]\n',
        '#![cfg(feature = "debug_assertions_helper")]\n',
    ):
        body = cfg_body(clean)
        got = {p for p in NON_FEATURE_PREDICATES if re.search(rf"\b{p}\b", body)}
        assert "debug_assertions" not in got, (
            f"false profile positive on {clean!r} — a substring match would "
            "make the PROFILE column unreadable"
        )

    # The default closure is TRANSITIVE, and stops at the crate boundary.
    feats = {
        "default": ["a"],
        "a": ["b", "dep:serde", "other/x"],
        "b": [],
        "off": [],
    }
    got = default_closure(feats)
    assert got == {"a", "b"}, f"default closure = {got}, want {{a, b}}"
    assert "off" not in got, "a non-default feature entered the closure"
    assert "other/x" not in got, "a dependency feature entered the OWN-crate closure"
    # Path resolution: a row with an explicit `path` must claim THAT file, not
    # the file whose stem matches its name. This is the false positive that
    # shipped in the first cut — it reported four already-guarded katgpt-rs
    # targets as defects, and "fixing" them added a duplicate target per file.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "a.goat.rs").write_text('#![cfg(feature = "x")]\n')
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\nversion = "0.0.0"\n\n'
            "[features]\nx = []\n\n"
            '[[test]]\nname = "a_goat"\npath = "tests/a.goat.rs"\n'
            'required-features = ["x"]\n'
        )
        r = RepoReport(repo="p")
        scan_manifest(root, root / "Cargo.toml", r)
        assert r.gated == 1, f"gate not seen: {r.gated}"
        assert r.covered == 1, "a path-declared, required-features row read as UNCOVERED"
        assert not r.findings, f"false positive: {[f.path for f in r.findings]}"

    # The three-way split of the non-feature class. `not(target_arch)` and
    # `target_arch` differ by one token and are OPPOSITE in severity, so a
    # classifier that pools them reports a number that means nothing — which is
    # what it did until Issue 713 T5 enumerated the 21 by hand.
    import tempfile as _tf

    for src, want in (
        ('#![cfg(not(target_arch = "wasm32"))]\n', "platform_except"),
        ('#![cfg(target_arch = "wasm32")]\n', "platform_only"),
        ("#![cfg(test)]\n", "cfg_test"),
    ):
        with _tf.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tests").mkdir()
            (root / "tests" / "t.rs").write_text(src)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "p"\nversion = "0.0.0"\n'
            )
            r = RepoReport(repo="p")
            scan_manifest(root, root / "Cargo.toml", r)
            got = {
                k: len(getattr(r, k))
                for k in ("platform_only", "platform_except", "cfg_test")
            }
            assert got[want] == 1 and sum(got.values()) == 1, f"{src!r} -> {got}"
            assert len(r.unexpressible) == 1, "the pooled property drifted"

    # Predicate detection must be TOKEN-based: a feature whose NAME contains a
    # predicate word must not be reported as carrying that predicate.
    assert not re.search(r"\btest\b", 'feature = "fastest_path"')

    # An empty/absent [features] table means NOTHING is default-on, so every
    # gated target is severe rather than latent. The wrong way round would
    # silently downgrade every finding in a crate with no feature table.
    assert default_closure({}) == set(), "empty feature table produced defaults"

    # The load-bearing classifier, BOTH directions. A false negative shrinks
    # the class that `cfg_gated_floor_gate.py` pins at zero — i.e. it turns
    # that gate into a permanent green — and a false positive from a substring
    # match ("aggregate" contains "gate") makes the class unreadable and so
    # ignored. Neither shows up in the totals.
    for name in (
        "plan414_hla_committed_belief_probe_goat.rs",
        "bench_256_kv_outer.goat.rs",     # the dotted dialect: `.` is a separator
        "test_g3_no_regression.rs",       # g<N> ordinal
        "seal_halt_drill.rs",
        "kv_conservation_check.rs",
        "feature_isolation_gate.rs",
        # The six the first cut got WRONG, found by diffing this classifier
        # against the substring grep that produced Issue 713's published
        # load-bearing table (87 vs 93). Every one is a real target; they are
        # pinned here because they are the shapes a "reasonable" token matcher
        # drops.
        "block_producer_g16f_cost.rs",      # G<N> with a variant suffix
        "kat_promotion_g2p.rs",
        "kat_stake_client_g2s.rs",
        "kat_vote_client_g9gov.rs",
        "t40_fixer_regate_harness.rs",      # a named compound, not a substring
        "prod_l3_sigkill_drills.rs",        # plural
        # The dialect the 2026-09-03 addition exists for — every one a real
        # workspace target that read as NOT load-bearing before it.
        "certified_frontier_correctness.rs",
        "bench_688_certified_frontier_alloc_check.rs",
        "hla_eigenbasis_determinism.rs",
        "kimi_k3_checkpoint_equivalence.rs",
        "merkle_soundness_spec_match.rs",
        "conformal_floor_bom.rs",
        "mla_backward_grad_check.rs",
    ):
        assert is_load_bearing(name), f"load-bearing name missed: {name}"
    for name in (
        "aggregate_stats.rs",             # contains "gate"
        "delegate_router.rs",
        "propagate_bounds.rs",
        "investigate_latency.rs",
        "mitigate_drift.rs",
        "g_probe.rs",                     # bare `g` is not a G<N> ordinal
        "spinning_up.rs",                 # contains "pin"
        "audition_pool.rs",               # contains "audit"
        # The 2026-09-03 tokens must stay TOKENS too: an allocator benchmark is
        # not an alloc gate, a flooring routine is not a floor gate, and a
        # gradient-descent driver is not a grad check.
        "allocator_pressure_bench.rs",    # contains "alloc"
        "flooring_math.rs",               # contains "floor"
        "gradient_descent_driver.rs",     # contains "grad"
        "determine_route.rs",             # contains "determin"
    ):
        assert not is_load_bearing(name), f"substring false positive: {name}"


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
        # Machine-readable, for `cfg_gated_floor_gate.py`. The consumer must
        # not re-derive any of this: a second copy of the classifier is a
        # second thing to keep in step, and the one that drifts is silently the
        # more permissive one.
        import json

        print(
            json.dumps(
                {
                    r.repo: {
                        "scanned": r.scanned,
                        "gated": r.gated,
                        "covered": r.covered,
                        "silent_now": len(r.silent_now()),
                        "silent_now_load_bearing": len(r.silent_now_load_bearing()),
                        "load_bearing_paths": sorted(
                            f.path for f in r.silent_now_load_bearing()
                        ),
                        "latent": len(r.silent_latent()),
                        "platform": len(r.unexpressible),
                        "platform_only": len(r.platform_only),
                        "platform_only_paths": sorted(f.path for f in r.platform_only),
                        "platform_except": len(r.platform_except),
                        "cfg_test": len(r.cfg_test),
                        "any_of": len(r.any_of),
                        # Additive, and OVERLAPPING: `profile_gated` is not a
                        # partition member, so existing consumers' sums are
                        # unchanged.
                        "profile_gated": len(r.profile),
                        "profile_gated_release_only": sum(
                            1 for f in r.profile if f.release_only
                        ),
                        "profile_gated_debug_only": sum(
                            1 for f in r.profile if not f.release_only
                        ),
                        "profile_gated_paths": sorted(f.path for f in r.profile),
                    }
                    for r in (audit(x) for x in repos)
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"cfg-gated target audit — {len(repos)} repo(s), population {scope}\n")
    header = (
        f"{'repo':<24} {'targets':>8} {'#![cfg]':>8} {'w/ req-f':>9} "
        f"{'SILENT-NOW':>11} {'load-bear':>10} {'latent':>7} {'plat-only':>10} "
        f"{'plat-exc':>9} {'cfg(test)':>10} {'any()':>6} {'PROFILE*':>9}"
    )
    print(header)
    print("-" * len(header))

    reports = [audit(r) for r in repos]
    total = 0
    latent = 0
    for rep in reports:
        total += len(rep.silent_now())
        latent += len(rep.silent_latent())
        print(
            f"{rep.repo:<24} {rep.scanned:>8} {rep.gated:>8} {rep.covered:>9} "
            f"{len(rep.silent_now()):>11} {len(rep.silent_now_load_bearing()):>10} "
            f"{len(rep.silent_latent()):>7} "
            f"{len(rep.platform_only):>10} {len(rep.platform_except):>9} "
            f"{len(rep.cfg_test):>10} {len(rep.any_of):>6} {len(rep.profile):>9}"
        )

    profile_total = sum(len(r.profile) for r in reports)
    profile_covered = sum(1 for r in reports for f in r.profile if f.declared)
    profile_lb = sum(1 for r in reports for f in r.profile if f.load_bearing)
    rel_only = [f for r in reports for f in r.profile if f.release_only]
    dbg_only = [f for r in reports for f in r.profile if not f.release_only]
    rel_lb = sum(1 for f in rel_only if f.load_bearing)
    dbg_lb = sum(1 for f in dbg_only if f.load_bearing)

    print(
        f"\nSILENT-NOW {total}: a plain `cargo test --test <name>` compiles the file to\n"
        f"nothing and prints `0 passed` with exit 0. latent {latent}: every gating\n"
        f"feature is default-on, so it only vanishes under `--no-default-features`.\n"
    )

    print(
        f"PROFILE* {profile_total} ({profile_lb} load-bearing): the `*` is because this\n"
        f"column OVERLAPS every other one — it is a second dimension, not a bucket, so\n"
        f"do not add it into the partition. A `#![cfg(..., not(debug_assertions))]` file\n"
        f"compiles to an EMPTY BINARY under `cargo test` — the default invocation, on the\n"
        f"right machine, with no flags typed. Every other predicate in this report needs\n"
        f"somebody to have CHOSEN the silent configuration (the wrong platform, miri,\n"
        f"`--no-default-features`); this one is what everybody gets by default.\n"
        f"It also SURVIVES the fix: {profile_covered} of the {profile_total} already have a\n"
        f"`required-features` row and so count as `w/ req-f`, which reads as protected.\n"
        f"It is not — the row makes NAMING the target honest, and cargo has no way to\n"
        f"express a profile at all. First measured by riir-ai `.issues/855` Class 2:\n"
        f"three riir-gpu targets, 0 errors in dev and 6/8/4 in release, with a planted\n"
        f"`compile_error!` failing to fire in dev.\n"
    )
    print(
        f"  Read the DIRECTION, never the pooled {profile_total} — the two halves are\n"
        f"  opposite, and differ by the single token `not(`:\n"
        f"    not(debug_assertions)  {len(rel_only):>4} ({rel_lb} load-bearing)  RELEASE-only:\n"
        f"        a green zero on plain `cargo test`, the default invocation.\n"
        f"    debug_assertions       {len(dbg_only):>4} ({dbg_lb} load-bearing)  DEBUG-only:\n"
        f"        runs by default, green zero under `--release` — the profile the perf\n"
        f"        rule mandates for gates, so these vanish exactly when someone follows it.\n"
    )
    for rep in reports:
        if not rep.profile:
            continue
        print(f"  {rep.repo}")
        for f in sorted(rep.profile, key=lambda x: (not x.release_only, x.path)):
            mark = "  [LOAD-BEARING]" if f.load_bearing else ""
            rf = "has req-f" if f.declared else "no [[%s]] row" % f.kind
            direction = "release-only" if f.release_only else "DEBUG-only"
            print(f"    [{direction}] {f.path}{mark}  ({rf})")
        print()
    if profile_total == 0:
        print("  (none)\n")
    for rep in reports:
        if not rep.silent_now():
            continue
        print(f"  {rep.repo}")
        for f in sorted(rep.silent_now(), key=lambda x: x.path):
            feats = ", ".join(f.features)
            plat = f" +[{', '.join(f.predicates)}]" if f.predicates else ""
            mark = "  [LOAD-BEARING]" if f.load_bearing else ""
            print(f"    {f.path}{mark}")
            print(f"      cfg: feature = {feats}{plat}  —  {f.reason}")
            print(
                f'      fix: [[{f.kind}]] / name = "{f.name}" / '
                f"required-features = {f.features!r}"
            )
        print()

    if total == 0:
        print("  (none)\n")

    print(
        "Report, not a gate — exit 0 always. The two non-defect classes above are\n"
        "shapes `required-features` CANNOT express (platform predicates; any-of\n"
        "feature sets, since cargo's required-features is AND-only). Reporting them\n"
        "apart is what keeps the SILENT column worth reading."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
