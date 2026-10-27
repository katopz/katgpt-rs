#!/usr/bin/env python3
"""
Cross-repo Cargo.toml inline-comment auditor.

Catches the drift class where a Cargo feature is transitively in
`default = [...]` (e.g. `micro_belief` is pulled in by `bom_sampling`,
which is in `default`), but the inline comment on the feature line
still claims "Opt-in" / "Default-OFF" / etc. Also catches the inverse
(comment claims default-on but feature is actually opt-in).

This is the Cargo.toml-comment counterpart to bench_doc_audit.py.
Run after every feature promotion to catch comment drift early.

Usage:
    python3 scripts/cargo_comment_audit.py                # audit this repo
    python3 scripts/cargo_comment_audit.py /git/riir-ai   # audit a specific repo
    python3 scripts/cargo_comment_audit.py /git           # walk all repos under /git

Exit code: 0 if no mismatches, 1 if any mismatch found.

Strategy
--------
1. Read every GIT-TRACKED `Cargo.toml` in the repo (falling back to a
   pruned filesystem walk when the target is not a git repo).
2. From each `[features]` table, collect:
   - the set of all defined feature names
   - the set of names transitively enabled by `default = [...]`
   (per-manifest closure, unioned across manifests)
3. For every feature definition line `feat = [...]  # comment`,
   parse the comment for a status phrase (opt-in / default-on / etc).
4. Cross-check: if comment says opt-in but feature is in the default
   closure → MISMATCH. If comment says default-on but feature is not
   in any default closure → MISMATCH.

Status vocabulary
-----------------
Reuses parse_status_phrase from bench_doc_audit — same word boundaries,
same opt-in-default-phrase guard (so "default-off" is not misclassified
as "default" via substring).

Caveats
-------
- Only the comment on the feature-definition line is checked. Multi-line
  comments above the feature line are NOT checked (would require a
  multi-line state machine; out of scope for v1).
- Only features defined in `[features]` tables are checked; package
  metadata comments are ignored.
- A feature is considered "default" if it appears in the default closure
  of ANY Cargo.toml in the repo (cross-crate promotions count).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

# Reuse the status parser from bench_doc_audit for consistency.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_doc_audit import (
    UNTRACKED_SKIPPED,
    audit_units,
    find_cargo_defaults,
    find_defined_features,
    iter_cargo_manifests,
    parse_status_phrase,
)

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()


def _parse_intra_crate_spec(spec: str) -> str | None:
    """Normalize a Cargo feature dep spec to a bare intra-crate feature name.

    Returns None for cross-crate specs (e.g. "katgpt-dec/foo") because those
    activate features in OTHER crates and don't expand THIS crate's closure.
    """
    if spec.startswith("dep:"):
        return None
    if "/" in spec:
        return None  # cross-crate spec — don't expand this crate's closure
    return spec or None


def find_cargo_defaults_per_manifest(repo_root: Path) -> dict[Path, set[str]]:
    """Per-manifest default-feature closure.

    Returns {cargo_path: set_of_features_in_default_closure}.
    Each manifest's closure is computed independently using only intra-crate
    feature specs.
    """
    result: dict[Path, set[str]] = {}
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if not feats:
            continue
        defaults: set[str] = set()
        deps: dict[str, set[str]] = {}
        for name, spec in feats.items():
            normalized: set[str] = set()
            if isinstance(spec, list):
                for s in spec:
                    n = _parse_intra_crate_spec(s)
                    if n:
                        normalized.add(n)
            deps[name] = normalized
            if name == "default":
                defaults |= normalized
        resolved = set(defaults)
        # ⚠ Issue 790 T3 read all three lines below against `arm_reach_audit`
        # and they are provably EQUIVALENT under mutation — recorded so the
        # next reader does not mistake "no arm kills it" for "untested":
        #   `len(deps) + 2`  a pure chain of N features plus `default` is N+1
        #                    entries needing N-1 rounds, and (N+1)-2 = N-1, so
        #                    even the slack-less bound is exactly sufficient.
        #   `changed = True/False`  flipping it only stops the early break; the
        #                    loop then runs its full bound and resolves the same.
        #   `not in resolved and not in new_added`  both targets are SETS, so
        #                    an `or` merely re-adds a member.
        # Each is a redundant guard, kept for legibility, and none can be
        # canaried. See AGENTS.md § An arm that exists and runs.
        changed = True
        for _ in range(len(deps) + 2):
            if not changed:
                break
            changed = False
            new_added: set[str] = set()
            for feat in list(resolved):
                for dep in deps.get(feat, ()):
                    if dep not in resolved and dep not in new_added:
                        new_added.add(dep)
            if new_added:
                resolved |= new_added
                changed = True
        result[cargo] = resolved
    return result


# Patterns indicating the comment is being precise about LOCAL scope.
# When present, the audit trusts the comment and does NOT flag a mismatch,
# even if a cross-crate union closure suggests the feature is default-on
# somewhere else in the workspace. Handles:
#   (a) "DEFAULT-ON in katgpt-dec (...); this root flag stays opt-in (...)"
#   (b) "Opt-in in katgpt-core (consumer enables transitively). Consumer
#       feature `X` PROMOTED to default-on (...)"
#   (c) "NOT in katgpt-core `default` — root's default-on forwarder activates it"
#       (explicit local-scope opt-in claim that overrides any later default-on
#       mention — the comment is being precise about which crate's default
#       does/doesn't include this feature).
LOCAL_SCOPE_OVERRIDE_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bstays\s+opt-in\b",
        r"\bstays\s+off\b",
        r"\bstays\s+OPT-IN\b",
        r"\bopt-in\s+in\s+[a-z][a-z0-9_-]+\b",
        r"\bopt-in\s+here\b",
        r"\bNOT\s+in\s+[a-z`][a-z0-9_-`]*\s+`?default`?\b",
        r"\bNOT\s+in\s+the\s+`?default`?\b",
    ]
]

# Cross-crate default-on claim patterns. When a default-on comment uses one
# of these patterns, the audit checks the UNION closure (across all crates)
# instead of the per-manifest closure. This is because the comment is making
# a claim about the feature's status in ANOTHER crate (typically the root
# crate or via a parent feature), not the current crate.
# Example: "DEFAULT-ON via rv_gated_routing" in katgpt-pruners/Cargo.toml
# is a claim about root's rv_gated_routing feature, not katgpt-pruners's own.
CROSS_CRATE_DEFAULT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bDEFAULT-ON\s+via\s+",
        r"\bdefault-ON\s+via\s+",
        r"\bdefault-on\s+via\s+",
        r"\bDEFAULT-ON\s+in\s+(?:root|katgpt)",
        r"\bdefault-on\s+in\s+(?:root|katgpt)",
    ]
]


# Match a feature definition line `name = [...]  # comment`.
# Cargo allows the RHS to span multiple lines, but inline comments on the
# same physical line as the feature name are the common case for one-liners
# like `micro_belief = ["dep:katgpt-micro-belief"]  # Opt-in until G1.1...`.
# For multi-line RHS, we capture only the comment on the first physical
# line (which is where the convention places the status phrase).
FEATURE_LINE_RE = re.compile(
    r"""^\s*
    (?P<name>[a-zA-Z][a-zA-Z0-9_-]*)    # feature name
    \s*=\s*
    \[                                  # opening bracket of feature list
    [^\n]*                              # rest of line (the spec list + maybe comment)
    $""",
    re.VERBOSE,
)


def extract_inline_comment(line: str) -> str | None:
    """Extract the `# ...` portion of a Cargo.toml line, respecting quotes.

    A naive `line.split("#", 1)[1]` would break on `#` inside quoted strings
    (rare but possible). We walk the line char-by-char tracking quote state.
    """
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(line):
        if in_quote:
            if ch == quote_char:
                in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            continue
        if ch == "#":
            return line[i + 1 :].strip()
    return None


# Demotion / negation patterns — explicit opt-in overrides.
# These beat any "default-on" mention because they represent an explicit
# decision to NOT promote (or to demote after promotion).
DEMOTE_NEGATION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdemoted\b",
        r"\bstays\s+opt-in\b",
        r"\bstays\s+off\b",
        r"\bnot\s+default-on\b",
        r"\bnot\s+promoted\b",
        r"\bpromotion\s+(?:blocked|deferred|pending|depends|waits)\b",
    ]
]

# Canonical DEFAULT-ON promotion template: "DEFAULT-ON (Plan/Issue/date...):".
# This is the strongest default-on signal — appears in fresh promotion commits
# and is unlikely to co-exist with stale opt-in language. The parens content
# MUST start with Plan/Issue/P\d+/date to avoid false matches on phrases like
# "default-on (behavior opt-in ...)" which describe a DIFFERENT feature.
CANONICAL_DEFAULT_RE = re.compile(
    r"DEFAULT-ON\s*\(\s*(?:Plan|Issue|P\d+|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

# Less-strict DEFAULT-ON mention (no parens, e.g. "DEFAULT-ON in root").
# Used after canonical and opt-in checks fall through.
DEFAULT_ON_MENTION_RE = re.compile(r"\bDEFAULT-ON\b", re.IGNORECASE)

# Explicit "Opt-in" / "OPT-IN" (capitalized emphasis) — human-written status
# claim. Deliberately NOT case-insensitive: bare lowercase "opt-in" is casual
# prose and is handled later by STRONG_OPTIN_RES, after the default-on checks.
#
# The all-caps form was missing, and it is a convention here, not a one-off:
# 32 comment lines use "OPT-IN" against 176 using "Opt-in". Any of those 32
# that also mention "default-on" incidentally — e.g. describing ANOTHER
# feature's promotion precedent — fell through rule 3 to rule 4's
# case-INSENSITIVE `\bDEFAULT-ON\b` and was classified default-on. That is how
# `signed_coupling_dynamics`, whose comment ends "OPT-IN — promotion waits on a
# production consumer", was reported as claiming default-on.
EXPLICIT_OPTIN_RE = re.compile(r"\b(?:Opt-in|OPT-IN)\b")

# Strong default-on phrases — fallback after canonical + Opt-in checks.
STRONG_DEFAULT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdefault-on\b",
        r"\bdefault on\b",
        r"\bon by default\b",
        r"\balways-on\b",
        r"\balways on\b",
        r"\bpromoted\b",
        r"\benabled by default\b",
    ]
]

# Strong opt-in phrases — fallback after explicit Opt-in check.
STRONG_OPTIN_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bopt-in\b",
        r"\bopt in\b",
        r"\bdefault-off\b",
        r"\bdefault off\b",
        r"\boff by default\b",
        r"\bdisabled by default\b",
        r"\bnot in default\b",
        r"\bnot default\b",
    ]
]

# Weak "default" — bare word, not part of a compound (default-features,
# default-off, default-on, default `value`).
# Negative lookahead: "default" NOT followed by `-`, `=`, or a parenthesised
# backticked VALUE in either order — "default (`0.82L`)" or "default `(0.82L)`".
#
# ⚑ Issue 789: the lookahead was `\s*\(` + backtick only, and the ONE live
# instance in the workspace is the other order — `default `(0.82L→0.45L)``, the
# exact string this comment has always quoted. The excluded shape appears in no
# manifest in any repo. It was latent because that line's comment also says
# "promotion blocked", so rung 1 reaches a verdict before rung 6 misfires; the
# fix changes no classification in the corpus (measured, 7465 comments).
#
# ⚠ Deliberately NOT widened to any backtick: `(?!\s*\(?`)` reads a
# code-reference `default` as no claim at all and takes 21 comments of the form
# "Not in `default` directly; transitively enabled via `X`" from `default` to
# `unknown` — silently dropping them from the cross-check. Those 21 do carry a
# real claim that no rung reads, which is its own gap and not this one's.
WEAK_DEFAULT_RE = re.compile(r"\bdefault\b(?![-=]|\s*(?:\(`|`\())", re.IGNORECASE)


def classify_comment(comment: str) -> tuple[str, str]:
    """Classify an inline comment to (status, raw_phrase).

    Precedence (strongest signal first):
      1. Demotion/negation patterns ("demoted", "stays opt-in", "not default-on",
         "promotion blocked/deferred/pending/depends") — opt-in wins because
         these represent an explicit decision to NOT promote.
      2. Canonical DEFAULT-ON promotion template ("DEFAULT-ON (Plan X...):") —
         strongest default-on signal, present in fresh promotion commits.
      3. Explicit "Opt-in" (capitalized emphasis) — human-written status claim.
      4. Other DEFAULT-ON mentions (no parens, e.g. "DEFAULT-ON in root").
      5. Strong default/opt-in phrases (fallback).
      6. Weak "default" — bare word, excluding compound forms.
      7. Unknown.

    Returns (status, raw_phrase) where raw_phrase is the matched substring
    for human-readable mismatch output.
    """
    if not comment:
        return "unknown", ""
    # 1. Demotion/negation overrides everything.
    for rx in DEMOTE_NEGATION_RES:
        m = rx.search(comment)
        if m:
            return "opt-in", m.group(0)
    # 2. Canonical promotion template.
    m = CANONICAL_DEFAULT_RE.search(comment)
    if m:
        return "default", "DEFAULT-ON template"
    # 3. Explicit capitalized "Opt-in".
    m = EXPLICIT_OPTIN_RE.search(comment)
    if m:
        return "opt-in", "explicit Opt-in"
    # 4. Other DEFAULT-ON mention (e.g. "DEFAULT-ON in root").
    m = DEFAULT_ON_MENTION_RE.search(comment)
    if m:
        return "default", m.group(0)
    # 5. Strong default phrases.
    for rx in STRONG_DEFAULT_RES:
        m = rx.search(comment)
        if m:
            return "default", m.group(0)
    # 5b. Strong opt-in phrases.
    for rx in STRONG_OPTIN_RES:
        m = rx.search(comment)
        if m:
            return "opt-in", m.group(0)
    # 6. Weak "default" — bare word.
    m = WEAK_DEFAULT_RE.search(comment)
    if m:
        return "default", m.group(0)
    return "unknown", ""


def selftest() -> list[str]:
    """Known-answer arms over the classifier ladder and the default closure.

    Issue 789. This gate was one of six docs_gate CHECKS whose failure path no
    test had ever executed — and it is the one with the most to lose from a
    silent regression, because `classify_comment` is a SEVEN-LEVEL precedence
    ladder whose order IS the verdict. A rule that moves one rung up or down
    does not error; it reclassifies comments, and both directions are wrong in
    a way that reads like a clean run.

    Every arm below is an input whose answer is decidable from the ladder's own
    docstring, and the ones marked ⚑ reproduce measured historical defects:
    the case-SENSITIVE `Opt-in` that missed this repo's 32 `OPT-IN` lines and
    mislabelled `signed_coupling_dynamics`, and the case-INSENSITIVE
    `default-off` substring that `parse_status_phrase`'s word boundaries fixed.

    Pure string work plus one temp manifest — runs unconditionally in `main`.
    """
    import tempfile

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    def status(comment: str) -> str:
        return classify_comment(comment)[0]

    # ── the ladder, rung by rung, in precedence order ─────────────────────
    # 1. Demotion/negation beats everything: an explicit decision NOT to
    #    promote outranks any incidental default-on mention.
    eq("rung 1 · demoted", status("DEFAULT-ON (Plan 9): demoted 2026-01-01"), "opt-in")
    eq("rung 1 · stays opt-in",
       status("default-on everywhere else; stays opt-in here"), "opt-in")
    eq("rung 1 · promotion deferred",
       status("DEFAULT-ON candidate; promotion deferred on GOAT G2"), "opt-in")
    eq("rung 1 · not promoted", status("default-on in root, not promoted here"), "opt-in")
    # 2. The canonical promotion template outranks a casual opt-in mention.
    eq("rung 2 · canonical template",
       status("DEFAULT-ON (Plan 340): supersedes the old opt-in path"), "default")
    # ⚑ The rung-2 paren guard is load-bearing only IN COMBINATION with rung 3:
    #   capitalised  -> rung 2 declines, rung 3 fires          -> opt-in
    #   lowercase    -> rung 2 declines, rung 3 misses, rung 4 -> default
    # Its own comment names the lowercase phrasing as the false match it
    # prevents, and that is not what happens. Both arms are pinned so the
    # asymmetry is a measurement rather than a surprise.
    eq("rung 2 · the paren guard defers to rung 3 (capitalised)",
       status("DEFAULT-ON (behavior Opt-in until a consumer lands)"), "opt-in")
    eq("rung 2 · lowercase falls through to rung 4, guard notwithstanding",
       status("DEFAULT-ON (behavior opt-in until a consumer lands)"), "default")
    # 3. ⚑ Explicit capitalised Opt-in, BOTH casings. The all-caps form was
    #    missing: 32 lines here use OPT-IN against 176 using Opt-in, and any
    #    that also mentioned default-on incidentally fell through to rung 4.
    eq("rung 3 ⚑ Opt-in", status("Opt-in — default-on precedent in riir-ai"), "opt-in")
    eq("rung 3 ⚑ OPT-IN (the measured miss)",
       status("OPT-IN — promotion waits on a production consumer"), "opt-in")
    eq("rung 3 ⚑ OPT-IN outranks a bare DEFAULT-ON mention",
       status("OPT-IN here; DEFAULT-ON in root"), "opt-in")
    # 4. A bare DEFAULT-ON mention, once rungs 1-3 have fallen through.
    eq("rung 4 · bare mention", status("DEFAULT-ON in root"), "default")
    # 5. Strong phrases, default before opt-in.
    eq("rung 5 · on by default", status("on by default since the GOAT gate"), "default")
    eq("rung 5 · promoted", status("promoted 2026-08-01"), "default")
    eq("rung 5b · off by default", status("off by default; enable for eval"), "opt-in")
    eq("rung 5b ⚑ default-off is not 'default' by substring",
       status("default-off until G3 passes"), "opt-in")
    eq("rung 5b · not in default", status("not in default"), "opt-in")
    # 6. Weak bare "default" — and the compound forms it must NOT fire on.
    eq("rung 6 · bare default", status("in the default set"), "default")
    eq("rung 6 · default-features is not a status claim",
       status("needs default-features = false"), "unknown")
    # ⚑ Both orders. Backtick-then-paren is the shape that actually occurs
    # (`cross_stage_relocation`, root Cargo.toml) and the one the lookahead
    # missed until Issue 789; paren-then-backtick is the shape it did exclude
    # and which occurs nowhere.
    eq("rung 6 · a default VALUE is not a status claim (backtick first)",
       status("default `(0.82L)` tuned per Plan 9"), "unknown")
    eq("rung 6 · a default VALUE is not a status claim (paren first)",
       status("default (`0.82L`) tuned per Plan 9"), "unknown")
    # A backticked code reference to the default ARRAY still reads as a claim
    # — pinned deliberately, because widening the lookahead to any backtick
    # drops 21 live comments out of the cross-check (see WEAK_DEFAULT_RE).
    eq("rung 6 · a backticked `default` reference still reads as a claim",
       status("Not in `default` directly; transitively enabled via `x`"), "default")
    # 7. Unknown, and the empty case.
    eq("rung 7 · no status phrase", status("tuning knob for the decoder"), "unknown")
    eq("rung 7 · empty comment", classify_comment(""), ("unknown", ""))

    # ── extract_inline_comment: a `#` inside a quoted string is not a comment
    eq("a plain inline comment", extract_inline_comment('foo = []  # Opt-in'), "Opt-in")
    eq("a # inside double quotes is not a comment",
       extract_inline_comment('foo = ["bar#baz"]'), None)
    eq("a # inside quotes does not shadow the real comment",
       extract_inline_comment('foo = ["bar#baz"]  # Opt-in'), "Opt-in")
    eq("no comment at all", extract_inline_comment("foo = []"), None)

    # ── _parse_intra_crate_spec: only intra-crate specs expand the closure ─
    eq("a bare feature expands the closure", _parse_intra_crate_spec("alpha"), "alpha")
    eq("a cross-crate spec does not",
       _parse_intra_crate_spec("katgpt-dec/alpha"), None)
    eq("an optional-dep spec does not", _parse_intra_crate_spec("dep:serde"), None)
    eq("the empty spec does not", _parse_intra_crate_spec(""), None)

    # ── find_cargo_defaults_per_manifest: the TRANSITIVE closure ───────────
    # The whole drift class this gate names: `micro_belief` is default-on only
    # because `bom_sampling` pulls it in. A one-hop closure misses it and the
    # gate then agrees with every stale "Opt-in" comment on such a feature.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "Cargo.toml").write_text(
            '[features]\n'
            'default = ["bom_sampling"]\n'
            'bom_sampling = ["micro_belief"]\n'
            'micro_belief = ["deep"]\n'
            'deep = []\n'
            'unrelated = []\n'
            'cross = ["katgpt-dec/other"]\n', encoding="utf-8")
        per = find_cargo_defaults_per_manifest(repo)
        closure = next(iter(per.values())) if per else set()
        eq("the closure is transitive, not one-hop",
           sorted(closure), ["bom_sampling", "deep", "micro_belief"])
        eq("an unrelated opt-in feature stays out", "unrelated" in closure, False)

        # A cycle must terminate rather than spin the fixed-point loop.
        (repo / "Cargo.toml").write_text(
            '[features]\n'
            'default = ["a"]\n'
            'a = ["b"]\n'
            'b = ["a"]\n', encoding="utf-8")
        cyc = next(iter(find_cargo_defaults_per_manifest(repo).values()))
        eq("a feature cycle terminates", sorted(cyc), ["a", "b"])

        # ⚑ Transitive DEPTH, added by Issue 790 T3. This arm exists to pin
        # that the closure reaches the last link of a chain, and it is worth
        # having — but it does NOT kill the `len(deps) + 2` mutant, and the
        # reason is a proof rather than a missing fixture:
        #
        #   a pure chain of N features plus `default` is N+1 entries in `deps`,
        #   and propagating from the default set takes N-1 rounds. The mutated
        #   bound `len(deps) - 2` is (N+1)-2 = N-1 — EXACTLY sufficient. Adding
        #   unrelated features grows `deps` without growing the depth, so the
        #   slack only widens. No feature graph discriminates the two.
        #
        # So that row is provably EQUIVALENT. Recorded here so the next person
        # does not spend the twenty minutes this took.
        (repo / "Cargo.toml").write_text(
            '[features]\n'
            'default = ["a"]\n'
            'a = ["b"]\n'
            'b = ["c"]\n'
            'c = ["d"]\n'
            'd = []\n', encoding="utf-8")
        chain = next(iter(find_cargo_defaults_per_manifest(repo).values()))
        eq("a pure chain resolves to its LAST link",
           sorted(chain), ["a", "b", "c", "d"])

        # A manifest with no [features] table contributes no closure at all,
        # rather than an empty one that then reads as "nothing is default".
        (repo / "Cargo.toml").write_text('[package]\nname = "x"\n',
                                         encoding="utf-8")
        eq("a manifest with no [features] is not in the closure map",
           find_cargo_defaults_per_manifest(repo), {})

    fails += _audit_repo_arms()
    return fails


def _audit_repo_arms() -> list[str]:
    """End-to-end arms over the VERDICT, on a fixture repo.

    Issue 790 T3: 19 of this module's survivors were in `audit_repo` and
    `iter_cargo_comment_labels`, and both take a repo path — so the whole
    verdict was armable the entire time and nothing did it. `classify_comment`
    having arms is not the same as the GATE having them: the classifier can be
    perfect while the scope choice, the dedup, the override or the population
    count is wrong, and every one of those fails silently.
    """
    import contextlib
    import io
    import tempfile

    fails: list[str] = []

    def run(files: dict[str, str]) -> tuple[int, str]:
        """(mismatches, output) for a synthetic repo.

        ⚠ Each fixture gets a `[package]` header, and that is LOAD-BEARING
        rather than cosmetic: `find_cargo_defaults` (the UNION closure) resolves
        through the package graph and returns an EMPTY set for a manifest with
        no package name, while `find_cargo_defaults_per_manifest` reads the same
        file correctly. Fixtures without it made three arms fail against
        perfectly good code — measured, and pinned as its own arm below, so that
        asymmetry is a documented property rather than a trap for the next
        person writing a fixture here.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, (rel, body) in enumerate(files.items()):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not body.lstrip().startswith("[package]"):
                    body = f'[package]\nname = "fixture{i}"\n' + body
                p.write_text(body, encoding="utf-8")
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                n = audit_repo(root)
            return n, sink.getvalue()

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    # The class this whole auditor exists for: a TRANSITIVELY default-on
    # feature still labelled Opt-in.
    n, out = run({"Cargo.toml":
                  '[features]\n'
                  'default = ["bom_sampling"]\n'
                  'bom_sampling = ["micro_belief"]\n'
                  'micro_belief = []  # Opt-in — stale label\n'})
    eq("a transitively default-on feature labelled Opt-in is a MISMATCH", n, 1)
    eq("…and the message names the scope it consulted",
       "per-manifest" in out or "OPT-IN but feature IS" in out, True)

    # The inverse.
    n, _ = run({"Cargo.toml":
                '[features]\n'
                'default = []\n'
                'alpha = []  # DEFAULT-ON (Plan 9): promoted\n'})
    eq("a DEFAULT-ON label on an opt-in feature is a MISMATCH", n, 1)

    # Both labels correct.
    n, out = run({"Cargo.toml":
                  '[features]\n'
                  'default = ["alpha"]\n'
                  'alpha = []  # DEFAULT-ON (Plan 9): promoted\n'
                  'beta = []   # Opt-in\n'})
    eq("correct labels are not a mismatch", n, 0)
    # The POPULATION is a separate claim from the verdict: a run that checked
    # nothing prints "0 mismatches" and reads exactly like a clean one.
    eq("both labelled lines were checked", "checked 2 inline comments" in out, True)

    # The local-scope override is TRUSTED — the comment is being precise.
    n, _ = run({"Cargo.toml":
                '[features]\n'
                'default = ["alpha"]\n'
                'alpha = []  # stays opt-in at root by design\n'})
    eq("a local-scope override is trusted, not flagged", n, 0)

    # The SCOPE CHOICE, both directions, which is the subtlest thing here and
    # had no arm: an opt-in claim is judged PER-MANIFEST (a sub-crate may
    # default the same name without contradicting it) and a default-on claim
    # is judged against the UNION (the comment may describe any crate).
    split = {
        "Cargo.toml": '[features]\n'
                      'default = []\n'
                      'alpha = []  # Opt-in at root\n',
        "crates/sub/Cargo.toml": '[features]\n'
                                 'default = ["alpha"]\n'
                                 'alpha = []  # DEFAULT-ON (Plan 9): here\n',
    }
    n, _ = run(split)
    eq("an opt-in claim is scoped PER-MANIFEST, not to the union", n, 0)
    union = {
        "Cargo.toml": '[features]\n'
                      'default = []\n'
                      'alpha = []  # DEFAULT-ON in the sub-crate\n',
        "crates/sub/Cargo.toml": '[features]\n'
                                 'default = ["alpha"]\n'
                                 'alpha = []\n',
    }
    n, _ = run(union)
    eq("a default-on claim is judged against the UNION", n, 0)

    # Lines that are not feature definitions must be skipped, and `default`
    # itself is not a feature to label.
    n, out = run({"Cargo.toml":
                  '# Opt-in\n'
                  '[features]  # Opt-in\n'
                  'default = ["alpha"]  # Opt-in — not a feature row\n'
                  'alpha = []\n'})
    eq("comment/table/default lines are not labelled rows", (n, "checked 0" in out),
       (0, True))

    # ⚑ The closure ASYMMETRY, pinned so it is a property and not a trap: the
    # UNION closure resolves through the package graph and sees nothing in a
    # manifest with no `[package] name`, while the per-manifest closure reads
    # the same file fine. Every real manifest here has a package section, so
    # this is latent — and it cost three arms before it was understood.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        body = '[features]\ndefault = ["alpha"]\nalpha = []\n'
        (root / "Cargo.toml").write_text(body, encoding="utf-8")
        from bench_doc_audit import find_cargo_defaults as _union
        eq("the UNION closure needs a package name", _union(root), set())
        eq("the PER-MANIFEST closure does not",
           sorted(next(iter(find_cargo_defaults_per_manifest(root).values()))),
           ["alpha"])
        (root / "Cargo.toml").write_text('[package]\nname = "p"\n' + body,
                                         encoding="utf-8")
        eq("…and with a package name the two agree", _union(root), {"alpha"})

    # A commented row whose name is not in [features] is not a row either.
    n, out = run({"Cargo.toml":
                  '[features]\n'
                  'default = []\n'
                  'alpha = []\n'
                  'ghost = []  # Opt-in\n'})
    eq("an undefined feature name is skipped",
       "checked 1 inline comments" in out, True)

    return fails


def iter_cargo_comment_labels(repo_root: Path):
    """Yield (cargo_path, rel_path, lineno, line, feature_name, raw_comment, parsed_status).

    cargo_path is the absolute Path to the Cargo.toml file (for per-manifest
    closure lookup); rel_path is the display path.
    raw_comment is the full inline comment text (for the caller to apply
    local-scope-override and cross-crate-claim checks).
    """
    for cargo in iter_cargo_manifests(repo_root):
        rel = cargo.relative_to(repo_root).as_posix()
        try:
            text = cargo.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if not feats:
            continue
        defined_names = set(feats.keys())
        for ln, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # ⚠ A fast SKIP, not a filter: the regex below already rejects a
            # blank line, a `#` comment and a `[table]` header, because it
            # anchors on `^[a-zA-Z]…=\s*\[`. So this line is provably
            # EQUIVALENT under mutation (Issue 790 T3 measured it surviving an
            # `or -> and` flip) and is kept for legibility and cost, not
            # correctness. The arms still pin the BEHAVIOUR — a comment line
            # and a table header must not be labelled rows.
            if not stripped or stripped.startswith("#") or stripped.startswith("["):
                continue
            m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*\[", stripped)
            if not m:
                continue
            name = m.group(1)
            if name not in defined_names:
                continue
            if name == "default":
                continue
            comment = extract_inline_comment(line)
            if not comment:
                continue
            parsed, _raw = classify_comment(comment)
            if parsed == "unknown":
                continue
            yield (cargo, rel, ln, line.rstrip(), name, comment, parsed)


def audit_repo(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    print(f"\n=== Auditing {repo_root.name} ===")
    per_manifest_defaults = find_cargo_defaults_per_manifest(repo_root)
    union_defaults = find_cargo_defaults(repo_root)
    defined_features = find_defined_features(repo_root)

    mismatches = 0
    checked = 0
    seen: set[tuple[str, int]] = set()
    for cargo, rel, ln, line, feat, comment, parsed in iter_cargo_comment_labels(
        repo_root
    ):
        if feat not in defined_features:
            continue
        if (rel, ln) in seen:
            continue
        seen.add((rel, ln))
        checked += 1
        # Local-scope override: comment contains "stays opt-in" / "stays off" /
        # "Opt-in in <crate>" / "Opt-in here" / "NOT in <crate> default" — the
        # comment is being precise about LOCAL scope. Trust it; don't flag.
        if any(rx.search(comment) for rx in LOCAL_SCOPE_OVERRIDE_RES):
            continue
        # Choose which closure to consult:
        # - Default-on claims: check union. The comment may describe status in
        #   any crate (root forward, sub-crate default, etc.). Per-manifest is
        #   too narrow here.
        # - Opt-in claims: check per-manifest. A root-level opt-in comment is
        #   precise about root's status; the feature may well be default-on in
        #   a sub-crate without contradicting the comment.
        if parsed == "default":
            defaults = union_defaults
            scope = "union"
        else:
            defaults = per_manifest_defaults.get(cargo, set())
            scope = "per-manifest"
        is_default = feat in defaults
        if parsed == "default" and not is_default:
            mismatches += 1
            print(
                f"  [MISMATCH] comment says DEFAULT-ON but feature NOT in any default closure"
            )
            print(f"    file: {rel}:{ln}  (checked {scope} closure)")
            print(f"    feat: {feat}")
            print(f"    line: {line}")
        elif parsed == "opt-in" and is_default:
            mismatches += 1
            print(
                f"  [MISMATCH] comment says OPT-IN but feature IS in this manifest's default closure"
            )
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {feat}")
            print(f"    line: {line}")
    # Untracked manifests are excluded from the closure (see
    # bench_doc_audit._tracked_manifests) — say so, rather than letting a
    # workstation run quietly differ from CI's count with no explanation.
    _untracked = UNTRACKED_SKIPPED.get(str(repo_root), 0)
    _tail = (f"  -> checked {checked} inline comments, {mismatches} mismatches"
             + (f" [skipped: {_untracked} untracked manifest(s) not in git]"
                if _untracked else ""))
    print(_tail)
    return mismatches


def main(argv: list[str]) -> int:
    # The canary first. `classify_comment`'s precedence ORDER is the verdict:
    # a rule that moves one rung reclassifies comments in both directions and
    # errors on nothing, so a failure here is not a finding — it means no
    # verdict is possible (exit 2, the house convention).
    arm_failures = selftest()
    if arm_failures:
        print("✗ INSTRUMENT: cargo_comment_audit's own selftest does not pass, so "
              "every classification below would be unreadable:")
        for f in arm_failures:
            print(f)
        return 2
    if len(argv) < 2:
        here = Path(__file__).resolve().parent.parent
        return 1 if audit_repo(here) else 0
    total = 0
    for arg in argv[1:]:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            print(f"skip (not a dir): {arg}", file=sys.stderr)
            continue
        for unit in audit_units(p):
            total += audit_repo(unit)
    print(f"\n=== TOTAL mismatches across all repos: {total} ===")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
