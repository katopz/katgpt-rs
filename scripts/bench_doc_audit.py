#!/usr/bin/env python3
"""
Cross-repo bench-doc label auditor.

Catches the drift class where a Cargo feature has been promoted to
`default = [...]` in a Phase N+1 commit, but the original benchmark doc
still labels it as `(opt-in)` / `(default-off)` / etc. Also catches the
inverse (doc claims default but feature is actually opt-in).

Re-runnable. Run after every feature promotion to catch label drift early.

Usage:
    python3 scripts/bench_doc_audit.py                # audit this repo
    python3 scripts/bench_doc_audit.py /git/riir-ai   # audit a specific repo
    python3 scripts/bench_doc_audit.py /git           # walk all repos under /git

Exit code: 0 if no mismatches, 1 if any mismatch found.

Strategy
--------
1. Read every GIT-TRACKED `Cargo.toml` in the repo (falling back to a
   pruned filesystem walk when the target is not a git repo).
2. From each `[features]` table, collect:
   - the set of all defined feature names
   - the set of names that appear in `default = [...]`
   (Names with `pkg/foo/` or `dep:foo` form are normalized to `foo`.)
3. Walk every `.benchmarks/*.md` and `.docs/*.md` looking for lines
   matching `Feature (gate|flag|s)?:`.
4. On each such line, find every `` `feature_name` (status-word) `` token.
5. If status-word parses to default/opt-in, cross-check against the
   Cargo-derived sets. Only feature names that actually exist as Cargo
   features are checked — this avoids false positives like `chain_curator`
   which is a use-case name, not a Cargo feature.

Status word vocabulary
----------------------
default-words: default, default-on, always-on, promoted, ...
opt-in-words:  opt-in, default-off, gated, optional, experimental, ...
"""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path

import tomllib

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

STATUS_WORDS_DEFAULT = {
    "default",
    "default-on",
    "default-on-since",
    "defaultfeature",
    "promoted",
    "promoted-to-default",
    "always-on",
}

STATUS_WORDS_OPTIN = {
    "opt-in",
    "optin",
    "off",
    "default-off",
    "gated",
    "feature-gated",
    "optional",
    "experimental",
    "disabled-by-default",
}

# Phrases that contain the substring "default" but mean OPT-IN.
# These must be checked before STATUS_WORDS_DEFAULT substrings, otherwise
# the bare "default" substring inside them would falsely trigger a default
# verdict (the session-13 parser bug).
OPTIN_DEFAULT_PHRASES = (
    "default-off",
    "off by default",
    "off-by-default",
    "not default",
    "not in default",
    "not promoted",
    "stays opt-in",
    "stays off",
)


class BlindRead(Exception):
    """A file that EXISTS and could not be READ — the instrument is blind.

    Measured (Issue 790 Finding 9, 2026-09-15). What motivated this was a
    docs-gate run on a resource-starved box where this audit printed
    `checked 97 labels, 56 mismatches` — a well-formed verdict, a plausible
    number, and completely wrong (0 on re-run). The issue INFERRED the
    mechanism and got it half wrong, so here is the measurement instead, one
    forced failure mode at a time against this repo's real tree:

    | forced failure              | verdict printed                    | exit |
    |-----------------------------|------------------------------------|------|
    | every `.md` read raises     | `checked 0 labels, 0 mismatches`   |  0   |
    | every manifest read raises  | `checked 0 labels, 0 mismatches`   |  0   |
    | 10% of manifest reads raise | `checked 97 labels, 1 mismatches`  |  1   |
    | 50% of manifest reads raise | `checked 97 labels, 24 mismatches` | 24   |

    Two corrections to the inference fall out of that. The manifest direction
    is NOT "the loud one": an empty default closure makes every label
    unresolvable rather than mismatched, so it prints the same confident green
    as the docs direction. And a label FLOOR — the repair the issue proposed —
    catches both TOTAL failures and would NOT have caught the run that was
    actually observed, because a PARTIAL read keeps the label count at its full
    97 and corrupts only the model those labels are judged against.

    So the floor is necessary and not sufficient, and this is the other half:
    an `OSError` on a file the walk just listed means the tree is unreadable,
    not that the manifest has no features. It ABORTS, with its own exit code —
    an instrument that cannot see must not return a number at all. A
    `TOMLDecodeError` is the opposite case (real content the audit should
    report and walk past) and keeps its warning.
    """


# `repo_root -> .md files WALKED` for the run just finished. The population
# behind the label count, and the only thing in the output that distinguishes
# "this repo has no feature labels" from "this instrument read nothing".
DOCS_WALKED: dict[str, int] = {}

# The label count of each `audit_repo` call, in order. Published for the same
# reason DOCS_WALKED is: the floor is asserted OUTSIDE the function that
# computes the number, so the number has to leave it.
LABELS_CHECKED: list[int] = []


def _parse_feature_spec(spec: str) -> str | None:
    """Normalize a Cargo feature dep spec to a bare feature name.

    Cargo feature dep specs come in several forms:
      "foo"                -> "foo"
      "dep:foo"            -> None  (dep activation, not a feature)
      "crate/foo"          -> "foo"
      "crate/foo/bar"      -> "foo"  (path-style)
      "pkg?/foo"           -> "foo"  (weak dep)
    """
    if spec.startswith("dep:"):
        return None
    # strip optional pkg prefix
    name = spec.split("/")[-1]
    return name or None


# Directories that never contain a source manifest but dominate the walk.
# `target` alone held 117 GB / ~1.3M entries in katgpt-rs when this was written.
PRUNE_DIRS = frozenset({"target", ".git", "node_modules", "__pycache__", ".venv"})


# repo root -> how many on-disk manifests were skipped as untracked. Reported,
# never silently dropped: this script's whole discipline is that a quietly
# smaller corpus still prints "0 mismatches".
UNTRACKED_SKIPPED: dict[str, int] = {}


def _tracked_manifests(repo_root: Path) -> list[Path] | None:
    """Manifests git actually tracks, or None if this is not a usable git repo.

    Why ask git at all: a workstation tree carries manifests CI will never see,
    and they do not merely pad a count — every one of them feeds the DEFAULT
    CLOSURE these audits are judged against, so an untracked copy can add a
    feature to the default set and flip a verdict. Measured 2026-09-01:
    riir-chain's `cloudflare/edge-wallet-container/.container-src/` is an
    untracked COPY OF THE REPO, and it made the local audit report 4 inline
    comments where CI reported 2. katgpt-rs walks 12 untracked manifests of 44;
    riir-train, 371 of 376.

    In CI this is a no-op by construction — a fresh checkout has nothing
    untracked — so it changes nothing there and makes the workstation agree.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*Cargo.toml"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None  # not a git repo, or git unavailable — fall back to the walk
    out = [repo_root / n.decode() for n in r.stdout.split(b"\0") if n]
    # An empty result from a real git repo is ambiguous (a repo genuinely
    # without manifests vs a broken invocation). Fall back rather than silently
    # audit nothing — the failure mode this whole family of scripts exists for.
    return out or None


def iter_cargo_manifests(repo_root: Path):
    """Every source Cargo.toml — git-tracked ones if this is a git repo, else a
    walk that prunes build/VCS dirs DURING traversal.

    The walk's previous form was `repo_root.rglob("Cargo.toml")` followed by
    `if "target" in cargo.parts: continue` — which filters AFTER pathlib has
    already descended into target/. Measured on katgpt-rs: 1,470,215 entries in
    10.7s unpruned versus 143,275 in 0.3s pruned, and four call sites each paid
    it once per run.
    """
    tracked = _tracked_manifests(repo_root)
    if tracked is not None:
        walked = sum(1 for _ in _walk_cargo_manifests(repo_root))
        UNTRACKED_SKIPPED[str(repo_root)] = max(0, walked - len(tracked))
        yield from tracked
        return
    yield from _walk_cargo_manifests(repo_root)


def _walk_cargo_manifests(repo_root: Path):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        if "Cargo.toml" in filenames:
            yield Path(dirpath) / "Cargo.toml"


def load_manifests(repo_root: Path) -> dict[str, dict]:
    """`package name -> features table` for every source manifest in the repo."""
    out: dict[str, dict] = {}
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except OSError as e:
            # NOT a manifest without features — a manifest this process could
            # not read. Every verdict below is judged against the default
            # closure these files build, so continuing here does not drop one
            # package, it silently re-bases the whole audit.
            raise BlindRead(f"could not READ {cargo}: {e}") from e
        except Exception as e:
            print(f"WARN: could not parse {cargo}: {e}", file=sys.stderr)
            continue
        feats = data.get("features") or {}
        pkg = (data.get("package") or {}).get("name")
        if pkg and feats:
            out.setdefault(pkg, {}).update(feats)
    return out


def find_cargo_defaults(repo_root: Path) -> set[str]:
    """Features a default build turns on anywhere in the repo.

    Reachability over a `(package, feature)` graph, from every member's
    `default` as a root. This replaces two heuristics that were each wrong in
    an opposite direction, and it is why the graph is worth the extra pass:

    * **Per-manifest closure UNDER-approximated.** It could not follow a chain
      that leaves a manifest and comes back:
      `riir-games-civ/default -> osc_emotion -> riir-games-shared/osc_emotion
      -> osc_npc`. `osc_npc` ships on in a default build and the old model
      reported it opt-in, so a doc stating the truth audited as drift.
    * **Collapsing `pkg/feat` to `feat` OVER-approximated.** riir-engine's
      default reaches `katgpt-core/tropical_algebra`; collapsing credited
      riir-engine's own, separately-defined `tropical_algebra` (which is NOT in
      its default). Here the edge simply leaves the repo and is not followed —
      measured 47 such bogus names in riir-ai alone.

    Walking `(pkg, feat)` nodes gets both right by construction rather than by
    two suppression rules layered on a wrong number.
    """
    return reachable_features(load_manifests(repo_root))


def reachable_nodes(mans: dict[str, dict]) -> set[tuple[str, str]]:
    """The `(package, feature)` reachability walk, as PAIRS — pure, so it can
    be pinned.

    The pair form is what the qualified-DEFAULT counterpart adjudicates on:
    "is the QUALIFIER's own flag on in a default build?" is a question about
    one NODE, not a bare name — `(riir-games-civ, mop_runtime)` and
    `(riir-engine, mop_runtime)` are different flags that happen to share a
    name, and only the node test tells an honest ancestor-forward apart from
    a same-name misattribution (see qualified_default_drift).
    """
    seen: set[tuple[str, str]] = {(p, "default") for p, f in mans.items()
                                  if "default" in f}
    stack = list(seen)
    while stack:
        pkg, feat = stack.pop()
        spec = mans.get(pkg, {}).get(feat)
        if not isinstance(spec, list):
            continue
        for x in spec:
            if not isinstance(x, str) or not x or x.startswith("dep:"):
                continue
            if "/" in x:
                tpkg, _, rest = x.partition("/")
                tpkg = tpkg.rstrip("?")
                tfeat = rest.split("/")[0]
                if tpkg not in mans:
                    continue  # edge leaves the repo — not ours to resolve
            else:
                tpkg, tfeat = pkg, x
            node = (tpkg, tfeat)
            if node not in seen:
                seen.add(node)
                stack.append(node)  # expanding an undefined feature is a no-op
    return seen


def reachable_features(mans: dict[str, dict]) -> set[str]:
    """Bare-name projection of `reachable_nodes` — the deployed model."""
    return {f for _, f in reachable_nodes(mans) if f != "default"}


def local_default_closure(feats: dict) -> set[str]:
    """Features one manifest turns on in its OWN default, via LOCAL edges only.

    `_parse_feature_spec` deliberately collapses `pkg/feat` to `feat`, which is
    right for the deployed model (the target crate is in this repo, so the flag
    really does turn on) and WRONG here: riir-engine has

        se2_equivariant = ["katgpt-core/tropical_algebra"]   # :503, in default
        tropical_algebra = ["katgpt-core/tropical_algebra", ...]  # :2085, NOT

    Collapsing the first entry credits riir-engine's own, separately-defined
    `tropical_algebra` as default-on. It is not, and the resulting report was a
    confident false drift on a doc that was correct. Only a BARE entry
    activates a feature of the same crate.
    """
    deps: dict[str, set[str]] = {}
    for name, spec in feats.items():
        acc: set[str] = set()
        if isinstance(spec, list):
            for x in spec:
                if x and not x.startswith("dep:") and "/" not in x:
                    acc.add(x)
        deps[name] = acc
    resolved = set(deps.get("default", ()))
    for _ in range(len(deps) + 2):
        add = {d for k in resolved for d in deps.get(k, ()) if d not in resolved}
        if not add:
            break
        resolved |= add
    # only a crate that DEFINES the feature can speak for its own default
    return {f for f in resolved if f in feats}


def own_closures_by_pkg(repo_root: Path) -> dict[str, set[str]]:
    """Per-package local default closures — the qualifier-scoped half of the
    forwarded-only rule.

    `find_own_crate_defaults` unions these; the union is the right model for a
    BARE label but the WRONG one for a qualified `pkg/feat` label when a
    DIFFERENT crate defines and defaults the same feature NAME: riir-ai's
    `riir-games-civ/mop_runtime` (opt-in) read as default-on because
    riir-engine — a different crate — promoted its own `mop_runtime` into
    `default` via `mop_psafe` (Plan 581 / Bench 906, the documented Defense-3
    layer split: engine default-on, game-layer forwards deliberately opt-in).
    The qualifier's crate is the one that speaks for its own flag.
    """
    out: dict[str, set[str]] = {}
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features") or {}
        if not feats:
            continue
        name = data.get("package", {}).get("name")
        if not name:
            continue
        out[name] = local_default_closure(feats)
    return out


def find_own_crate_defaults(repo_root: Path) -> set[str]:
    """Features that some crate turns on in its OWN default closure, counting
    only crates that actually DEFINE the feature.

    This is the strict subset of `find_cargo_defaults`. The difference between
    the two is the *forwarded-only* set: a feature whose owning crate ships it
    off, which is nonetheless on in a default build because an ancestor's
    `default` list names `child/feature`.

    Both readings of such a flag are defensible, so a doc that calls it opt-in
    is not drift. Measured 2026-09-01: riir-chain's
    `` `riir-wallet/siwr` (client + RP kit, default-OFF) `` is exactly this —
    off in riir-wallet, on via riir-chaind's `default -> chain_siwr`. Reporting
    it would have been a false positive on a doc that is right, which is the
    failure mode this script's own comments call out as how a gate earns a
    reputation for noise.
    """
    own: set[str] = set()
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features") or {}
        if not feats:
            continue
        # LOCAL activations only. `_parse_feature_spec` deliberately collapses
        # `pkg/feat` to `feat`, which is right for the deployed model (the
        # target crate is in this repo, so the flag really does turn on) and
        # WRONG here: `se2_equivariant = ["katgpt-core/tropical_algebra"]` does
        # not turn on riir-engine's own, separately-defined `tropical_algebra`
        # (riir-engine/Cargo.toml:503 vs :2085). Crediting it produced a
        # confident false drift report on a doc that was correct. Only a bare
        # entry activates a feature of the same crate.
        own |= local_default_closure(feats)
    return own


def qualified_optin_is_scoped(qual: str | None, feat: str,
                              own_by_pkg: dict[str, set[str]]) -> bool:
    """Does a qualified opt-in label's claim hold against ITS OWN crate?

    A `pkg/feat` label claims the QUALIFIER ships the flag off. That claim is
    falsified only by the qualifier's own default closure — not by another
    crate defaulting a same-named feature (the Defense-3 layer split: engine
    default-on + game-layer forward opt-in is deliberate, not drift — the
    riir-ai 681/905 lesson). BARE labels stay deployed-level claims: the
    union model still governs them, unchanged.

    Note the counterpart, handled since 2026-09-12: a QUALIFIED label
    claiming DEFAULT is adjudicated by `qualified_default_drift` against the
    qualifier's own closure AND its own `(pkg, feat)` node. It was landed
    non-blind: zero such labels existed at handling time (measured
    2026-09-12), so the tightening changed no verdict and its policy is
    pinned by selftest arms in both directions.
    """
    return qual is not None and feat not in own_by_pkg.get(qual, set())


def qualified_default_drift(qual: str | None, feat: str,
                             own_by_pkg: dict[str, set[str]],
                             defined_by_pkg: dict[str, set[str]],
                             deployed_nodes: set[tuple[str, str]]) -> bool:
    """Does a qualified DEFAULT label misattribute ANOTHER crate's default
    to the qualifier?

    The false-negative mirror of `qualified_optin_is_scoped` (which closed
    the false-positive direction, 13dbba6b). A `pkg/feat (DEFAULT...)`
    label used to pass via the bare-name union whenever ANY crate defaulted
    the NAME — including when the qualifier itself defines the flag, ships
    it OFF, and the on-ness belongs to a DIFFERENT crate's same-named flag.
    That is exactly the Defense-3 layer split read backwards: a doc claiming
    `riir-games-civ/mop_runtime` is default because riir-engine promoted ITS
    `mop_runtime` is attributing one crate's gate state to another.

    Fires ONLY when all three hold:
      1. the label is qualified and the qualifier DEFINES the flag (a
         qualifier that does not define it is citing a dependency's flag —
         the union model already adjudicated that claim);
      2. the qualifier's own default closure does NOT turn it on;
      3. the qualifier's own `(qual, feat)` node is NOT reachable from any
         default — the discriminator between the two sub-shapes. When the
         node IS reachable (riir-chaind's default forwards
         `riir-wallet/siwr`, the 2026-09-01 forwarded-only doctrine), the
         qualifier's own flag genuinely ships on in a default build and BOTH
         readings are defensible: counted by the caller, never flagged.
    """
    if qual is None or feat not in defined_by_pkg.get(qual, set()):
        return False
    if feat in own_by_pkg.get(qual, set()):
        return False  # the qualifier's own default turns it on — claim holds
    return (qual, feat) not in deployed_nodes


def find_package_names(repo_root: Path) -> set[str]:
    """Every `[package] name` in the repo — the set of qualifiers this repo can
    actually resolve.

    A doc label `riir-neuron-db/density_aware_wake` inside riir-ai names a
    SIBLING repo's flag. Checking it against riir-ai's own Cargo manifests is
    not a weaker check, it is a wrong one: measured 2026-09-01, 3 of the 12
    cross-repo tokens in the workspace carry a feature name that is ALSO
    defined locally with a different default status, so the naive read invents
    a mismatch on a doc that is correct. Siblings are also absent in CI, so
    resolving them there is not an option either — the qualifier is the only
    signal available, and an unresolvable one must be reported, not dropped.
    """
    names: set[str] = set()
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        name = (data.get("package") or {}).get("name")
        if name:
            names.add(name)
    return names


def find_defined_features(repo_root: Path) -> set[str]:
    """All feature names defined in any Cargo.toml under repo_root."""
    defined: set[str] = set()
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if feats:
            defined.update(feats.keys())
    return defined


FEATURE_HEADER_RE = re.compile(
    r"^\s*\**\s*Feature(?:s)?(?:\s+(?:gate|flag|flags|gates))?\**\s*[:\-]",
    re.IGNORECASE,
)

# After a Feature header, find each `feature_name` token optionally followed
# by a status indicator. Status indicators come in many forms:
#   `foo` (opt-in)                  — paren with status word
#   `foo` (**DEFAULT-ON** ...)       — paren with bolded status
#   `foo` = [...] (opt-in, NOT ...)  — Cargo-like syntax + paren
#   `foo` — OPT-IN, GOAT-gated       — em/en-dash separator
#   `foo` (opt-in, same as Phase 1) — paren with extra context
# We capture the feature name and a window of ~80 chars after it for status
# classification (instead of requiring the status to be in tight parens).
#   `crate/foo` (opt-in)            — namespaced (Cargo `pkg/feature` syntax)
# Groups are NAMED, not numbered: adding the qualifier group shifted every
# positional index, and a positional consumer would have silently started
# reading the qualifier as the feature name.
FEATURE_TOKEN_RE = re.compile(
    r"`(?:(?P<qual>[a-zA-Z][a-zA-Z0-9_\-]*)/)?"
    r"(?P<feat>[a-zA-Z][a-zA-Z0-9_\-]*?)`"
    r"(?:\s*=\s*\[[^\]]*\])?"  # optional Cargo-like = [...]
    r"\s*"  # optional whitespace
    r"(?:[\(\[]|\u2014|\u2013|--|:)"  # status separator: ( [ — – -- :
    r"?"  # separator optional (status may be inline)
    r"\s*\**"  # optional opening bold
    r"(?P<status>[a-zA-Z][a-zA-Z0-9_\-\s,+/]{0,80}?)"  # status phrase
    r"\**\s*"  # optional closing bold
    # Status boundary. `(` and `*` were missing, and their absence silently
    # dropped an entire sibling's convention: riir-chain writes
    #   **Feature:** `chain_block_producer` — **default-OFF** (implies ...)
    # where the status is followed by `**` then ` (`. Neither was a boundary,
    # the lazy status group could not terminate, and the line yielded NO token
    # at all — so 26 benchmark docs audited as "0 labels, 0 mismatches", which
    # reads identical to clean. Measured 2026-09-01: 9 repos / 62 docs in that
    # state (Issue 702).
    r"(?=[\)\]\n.,;:|(*]|$)",  # lookahead: status boundary
)


def parse_status_phrase(phrase: str) -> str:
    """Normalize a status phrase to 'default' | 'opt-in' | 'unknown'.

    Uses word-boundary matching (not bare substring) so that phrases like
    "default-off" or "off by default" are correctly classified as opt-in,
    not as default. Also recognizes explicit opt-in/default-on overrides.
    """
    w = phrase.lower().strip().rstrip(".")
    if not w:
        return "unknown"
    # Opt-in phrases that contain the substring "default" must be checked
    # FIRST. Without this guard, "default-off" would match the default-word
    # "default" via substring and flip to default-on (the session-13 bug).
    for p in OPTIN_DEFAULT_PHRASES:
        if p in w:
            return "opt-in"
    # Word-boundary check for each default-word.
    # Use regex with word boundaries so "default" doesn't match inside
    # "default-off" (already handled above) or "not-default-on".
    has_default = any(
        re.search(r"\b" + re.escape(s) + r"\b", w) for s in STATUS_WORDS_DEFAULT
    )
    has_optin = any(
        re.search(r"\b" + re.escape(s) + r"\b", w) for s in STATUS_WORDS_OPTIN
    )
    has_not = " not " in f" {w} " or w.startswith("not ") or "!" in w
    # "opt-in, NOT default-on" pattern → opt-in wins.
    if has_optin and has_default and has_not:
        return "opt-in"
    if has_default and not has_not:
        return "default"
    if has_optin:
        return "opt-in"
    return "unknown"


# Keep old name as alias for backward compatibility / external callers.
def parse_status_word(word: str) -> str:
    return parse_status_phrase(word)


# A label may record its own history: "(opt-in at time of writing, 2026-06-30;
# **promoted to DEFAULT-ON 2026-07-20 by Plan 468**)". Reading only the FIRST
# status word flags that as a mismatch, which is backwards — the doc is doing
# exactly what it should, and the auditor was penalising it for being thorough.
# False positives on correct docs are how a gate earns a reputation for noise
# and gets ignored.
#
# So the LAST explicit transition on the line wins. Note this OVERRIDES rather
# than SUPPRESSES: a doc claiming "promoted to DEFAULT-ON" for a feature that
# is not in any default array now fails on the other branch, which a plain
# skip-if-promotion-mentioned rule would have hidden.
TRANSITION_RES = [
    (re.compile(r"(?:re-)?promoted\s+(?:back\s+)?to\s+\**default(?:-on)?\**", re.I),
     "default"),
    (re.compile(r"\bnow\s+\**default-on\**", re.I), "default"),
    # A feature with no bare entry in any `default[]` that a default build still
    # enables through a `pkg/feat` chain. This is a distinct third state, not a
    # sloppy way of saying default-on, and the workspace already names it:
    # riir-neuron-db's README has a §"Transitive default" section for it. Two
    # katgpt-rs benchmark docs are in exactly this state and said "opt-in".
    # Deliberately the FULL phrase, not a looser `transitive.*default`. The
    # negation guard below only inspects text immediately before the match, so a
    # pattern that can start mid-phrase escapes it: on
    # "(opt-in, NOT on by transitive default)" the loose form matched at
    # "transitive", six characters past the "NOT", and read the label as
    # default-on. The selftest's negation case caught it.
    (re.compile(r"\bon\s+by\s+transitive\s+default\b", re.I), "default"),
    (re.compile(r"\bdemoted\s+(?:back\s+)?to\s+\**opt-in\**", re.I), "opt-in"),
    (re.compile(r"\breverted\s+to\s+\**opt-in\**", re.I), "opt-in"),
    (re.compile(r"\bnow\s+\**opt-in\**", re.I), "opt-in"),
]


# "**NOT promoted to default — D2F is opt-in research**" contains the exact
# substring "promoted to default". The first version of the transition rule read
# that as a promotion and turned one false positive into two, on two docs that
# were stating their status correctly and emphatically. Negation is checked
# immediately before the phrase; `**` and whitespace may intervene.
NEGATION_RE = re.compile(r"\b(?:not|never|no|nor|without)\b[\s*_]*$", re.I)


def parse_terminal_transition(line: str, from_idx: int) -> tuple[str, str] | None:
    """Last explicit, non-negated status transition at/after from_idx."""
    best_at, best = -1, None
    for rx, status in TRANSITION_RES:
        for m in rx.finditer(line, from_idx):
            if NEGATION_RE.search(line[max(0, m.start() - 24):m.start()]):
                continue
            if m.start() > best_at:
                best_at, best = m.start(), (status, m.group(0))
    return best


# The captured status is a LAZY match ending at the first boundary, so a
# compound parenthetical loses everything after its first comma:
#
#   **Features:** `riir-wallet/siwr` (client + RP kit, default-OFF), forwarded
#
# captures "client + RP kit" -> unknown, and the real status word sits one
# comma further on. Retrying over the rest of the enclosing clause recovers it.
#
# This runs ONLY when the tight capture already parsed to `unknown`, i.e. on
# labels the caller was about to discard. It is therefore monotone by
# construction: it can promote unknown -> default/opt-in, and can never change
# a verdict the tokenizer already reached. That property is why it needs no
# A/B to be safe — but it was measured anyway (Issue 702).
CLAUSE_END_RE = re.compile(r"[\)\]|]")


def widened_status(line: str, from_idx: int) -> tuple[str, str] | None:
    """Re-read the status over the remainder of the enclosing clause."""
    end = CLAUSE_END_RE.search(line, from_idx)
    seg = line[from_idx : end.start() if end else min(len(line), from_idx + 160)]
    if not seg.strip():
        return None
    parsed = parse_status_phrase(seg)
    return None if parsed == "unknown" else (parsed, seg.strip())


# A label may explicitly scope its claim to one crate: "opt-in in this crate —
# see npc.md §Feature gate landscape for the layered gate split". That doc is
# describing a layered split it is fully aware of; the flat repo-wide model
# cannot adjudicate it and flagging it is noise. Counted, never silent.
SCOPED_CLAIM_RE = re.compile(r"\bin\s+(?:this|the)\s+crate\b", re.I)


def classify_token(line: str, m: "re.Match") -> tuple[str, str]:
    """(parsed_status, raw) for one token match — the ONE status path.

    The selftest previously re-implemented this inline, so a canary that broke
    the iterator's copy left the selftest green: the pin guarded a duplicate,
    not the production path. Both call this now.
    """
    raw = m.group("status")
    parsed = parse_status_phrase(raw)
    if parsed == "unknown":
        wide = widened_status(line, m.start("status"))
        if wide is not None:
            parsed, raw = wide[0], f"{raw!r} -> widened: {wide[1]!r}"
    # The terminal transition overrides whatever the base phrase said, and it
    # lives HERE rather than in the caller for the same reason the widening
    # does: a status rule outside the shared path is a rule the selftest cannot
    # reach. Adding the transitive-default vocabulary with this still in
    # `iter_bench_doc_labels` made the new pin fail while the production path
    # was correct — the pin was right, the factoring was wrong.
    trans = parse_terminal_transition(line, m.end())
    if trans is not None:
        parsed = trans[0]
        raw = f"{raw} | transition: {trans[1]!r}"
    return parsed, raw


def iter_bench_doc_labels(repo_root: Path):
    """Yield (rel, lineno, line, qualifier|None, feature, raw, parsed).

    Records the WALK size in `DOCS_WALKED` as it goes: a label count with no
    population under it cannot tell a repo that has no labels from a walk that
    found no files, and those print the same line.
    """
    walked = 0
    DOCS_WALKED[str(repo_root)] = 0
    for sub in (".benchmarks", ".docs"):
        d = repo_root / sub
        if not d.is_dir():
            continue
        for md in d.rglob("*.md"):
            rel = md.relative_to(repo_root).as_posix()
            walked += 1
            DOCS_WALKED[str(repo_root)] = walked
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                raise BlindRead(f"could not READ {rel}: {e}") from e
            except Exception:
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                if not FEATURE_HEADER_RE.search(line):
                    continue
                for m in FEATURE_TOKEN_RE.finditer(line):
                    qual = m.group("qual")
                    feat = m.group("feat")
                    parsed, raw = classify_token(line, m)
                    if parsed == "unknown":
                        continue
                    yield (rel, ln, line.strip(), qual, feat, raw, parsed)
    DOCS_WALKED[str(repo_root)] = walked


def pure_rule_arms() -> list[str]:
    """Arms over the PURE rules `selftest` and `TOKENIZER_CASES` step past.

    Issue 790 T2. The arm-reach audit found 23 live survivors in this module
    after T3, and the prior read filed them all as EQUIVALENT / I/O shell /
    message formatting. **That was too generous**: a third of them are plain
    functions over plain data, with no fixture repo and no subprocess between
    an arm and the decision. They are armed here.

    ⚠ One row that IS equivalent is kept out and proved rather than armed:
    `local_default_closure`'s `range(len(deps) + 2)`. For a chain seeded from
    `default`, the iterations NEEDED are the longest path from the seed, and
    `len(deps)` already counts `default` plus every node on it — so
    `len(deps) - 2` is still exactly sufficient on every graph that terminates.
    T3 reached the same conclusion by construction and could not build a
    discriminating fixture; the pin file records the proof rather than an arm.
    """
    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    # ── `return name or None`: an `and` here returns None for every REAL
    # feature, so every qualified spec would vanish and the audit would print
    # a smaller label count with "0 mismatches" — clean-looking.
    eq("_parse_feature_spec: qualified spec keeps the feature",
       _parse_feature_spec("pkg/feat"), "feat")
    eq("_parse_feature_spec: empty tail is None, not ''",
       _parse_feature_spec("pkg/"), None)

    # ── `has_not`'s three disjuncts. A fixture must make exactly ONE true, or
    # `and` and `or` agree and the arm is degenerate — the shape T3 found six
    # of. `!default-on` trips only the `"!" in w` arm.
    eq("parse_status_phrase: bare `!` negates",
       parse_status_phrase("!default-on"), "unknown")
    eq("parse_status_phrase: leading `not ` negates",
       parse_status_phrase("not default-on"), "opt-in")
    eq("parse_status_phrase: unnegated default-on still reads default",
       parse_status_phrase("default-on"), "default")

    # ── the closure walk. `default -> a -> b` with every hop DEFINED as a
    # feature: a 1-hop chain cannot distinguish the `not in resolved` filter
    # from its flip, and a hop whose target is not itself a key is dropped by
    # the final `f in feats` filter and discriminates nothing either.
    chain = {"default": ["a"], "a": ["b"], "b": []}
    eq("local_default_closure: a two-hop chain resolves transitively",
       sorted(local_default_closure(chain)), ["a", "b"])
    # `dep:` activations and `pkg/feat` forwards are NOT local defaults. With
    # `or` flipped to `and` in the filter, both would be admitted.
    eq("local_default_closure: dep: and pkg/ entries are not local",
       sorted(local_default_closure(
           {"default": ["dep:x", "other/y", "a"], "a": []})), ["a"])

    # ── `reachable_nodes`' skip guard, same shape one function over: with
    # `and`, a `dep:` entry is a real node and the graph grows phantom nodes.
    eq("reachable_nodes: a dep: activation is not a node",
       sorted(reachable_nodes({"p": {"default": ["dep:q", "f"], "f": []}})),
       [("p", "default"), ("p", "f")])

    # ── the scan itself IS armed (the later transition wins), but its `>` is
    # deliberately NOT: `>` and `>=` differ only on a TIE, and no two
    # TRANSITION_RES can match at one index — the only shared prefix is
    # `\bnow\s+`, whose two tails (`default-on` / `opt-in`) are mutually
    # exclusive. Proved, not armed; the pin file carries the argument.
    eq("parse_terminal_transition: the LATER transition wins",
       parse_terminal_transition("now default-on, and now opt-in", 0)[0],
       "opt-in")

    # ── `widened_status`' 160-char fallback window. A line with no clause end
    # must still yield the status AFTER from_idx; a flipped `+` makes the
    # slice empty and the function returns None, silently dropping the label.
    long_line = "x" * 40 + "default-on" + "y" * 5
    got = widened_status(long_line, 40)
    eq("widened_status: the no-clause-end window reaches forward",
       got[0] if got else None, "default")

    return fails


def manifest_reader_arms() -> list[str]:
    """Arms over the three functions that READ manifests into the models.

    Issue 790 T2. `audit_repo_arms` builds fixture repos and reaches the
    verdict; `selftest` feeds the closure functions hand-written dicts. Between
    them sat the readers — `load_manifests`, `own_closures_by_pkg`,
    `find_own_crate_defaults` — whose guards decide which packages enter the
    model at all. Every one of them fails in the QUIET direction: a package
    silently dropped here shrinks `checked` and still prints "0 mismatches".

    The fixture is two packages, and BOTH halves are load-bearing: one with
    features and one without. With only the first, `if not feats` and
    `if not name` flip harmlessly; with only the second, `feats = ... or {}`
    does.
    """
    import tempfile

    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "withfeat").mkdir()
        (root / "withfeat" / "Cargo.toml").write_text(
            '[package]\nname = "withfeat"\n'
            '[features]\ndefault = ["a"]\na = []\n', encoding="utf-8")
        (root / "nofeat").mkdir()
        (root / "nofeat" / "Cargo.toml").write_text(
            '[package]\nname = "nofeat"\n', encoding="utf-8")

        mans = load_manifests(root)
        if sorted(mans) != ["withfeat"]:
            fails.append(
                f"    load_manifests: got {sorted(mans)}, want ['withfeat'] — "
                f"a package with a name and NO features must not enter the "
                f"model as an empty feature table")
        own = own_closures_by_pkg(root)
        if own != {"withfeat": {"a"}}:
            fails.append(
                f"    own_closures_by_pkg: got {own}, want "
                f"{{'withfeat': {{'a'}}}}")
        if find_own_crate_defaults(root) != {"a"}:
            fails.append(
                f"    find_own_crate_defaults: got "
                f"{find_own_crate_defaults(root)}, want {{'a'}}")

        # A package with NO `[package] name` must be skipped, not credited to
        # the empty string. ⚠ Added as its own arm because the two flips of
        # `if not name` and `if not feats` are otherwise indistinguishable on
        # a fixture where every package has both.
        (root / "anon").mkdir()
        (root / "anon" / "Cargo.toml").write_text(
            '[features]\ndefault = ["zz"]\nzz = []\n', encoding="utf-8")
        if "" in own_closures_by_pkg(root):
            fails.append(
                "    own_closures_by_pkg: a manifest with no [package] name "
                "was credited under the empty string")
        # ── `_tracked_manifests`' two FALLBACK returns (Issue 790 T2). Both
        # return None to mean "ask the filesystem walk instead", and the
        # distinction from `[]` is the whole point: `[]` means AUDIT NOTHING,
        # which prints "0 mismatches" over an empty population. The guards sit
        # behind a `git` invocation, but the invocation is cheap and both
        # answers are reachable — a non-repo for the returncode arm, a repo
        # with no manifests for the empty-result arm.
        import subprocess
        nogit = root / "notarepo"
        nogit.mkdir()
        if _tracked_manifests(nogit) is not None:
            fails.append(
                "    _tracked_manifests: a non-git directory did not fall "
                "back — a nonzero `git` exit must yield None, never a list")
        empty = root / "emptyrepo"
        empty.mkdir()
        subprocess.run(["git", "-C", str(empty), "init", "-q"],
                       capture_output=True)
        if _tracked_manifests(empty) is not None:
            fails.append(
                "    _tracked_manifests: a git repo with NO manifests "
                "returned a list — an empty result is ambiguous (a repo "
                "genuinely without manifests vs a broken invocation) and must "
                "fall back rather than silently audit nothing")
        # ⚠ The SUCCESS case is a separate arm and both above are satisfied
        # without it: the two fallbacks each return None, so a `returncode
        # != 0` flipped to `== 0` still answers None to both — it just answers
        # None to EVERYTHING, and the audit quietly reverts to the 10.7s
        # unpruned walk on every repo. Only a repo that HAS a tracked manifest
        # distinguishes them.
        live = root / "liverepo"
        live.mkdir()
        subprocess.run(["git", "-C", str(live), "init", "-q"],
                       capture_output=True)
        (live / "Cargo.toml").write_text(
            '[package]\nname = "live"\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(live), "add", "Cargo.toml"],
                       capture_output=True)
        got = _tracked_manifests(live)
        if got is None or [p.name for p in got] != ["Cargo.toml"]:
            fails.append(
                f"    _tracked_manifests: a git repo WITH a tracked manifest "
                f"returned {got!r} — git tracking is the fast path and "
                f"falling back on success is silent, not wrong-looking")
        # …and the untracked-skipped DIAGNOSTIC counts the difference. An
        # untracked manifest is deliberately NOT audited; the count is how a
        # reader learns the population was smaller than the tree.
        (live / "sub").mkdir()
        (live / "sub" / "Cargo.toml").write_text(
            '[package]\nname = "untracked"\n', encoding="utf-8")
        list(iter_cargo_manifests(live))
        if UNTRACKED_SKIPPED.get(str(live)) != 1:
            fails.append(
                f"    iter_cargo_manifests: UNTRACKED_SKIPPED is "
                f"{UNTRACKED_SKIPPED.get(str(live))!r}, want 1 — the "
                f"diagnostic that tells a reader the audited population was "
                f"smaller than the tree")

        # ⚠ `find_own_crate_defaults` is deliberately NOT asserted on this
        # input. It has no name guard at all, so it DOES union an unnamed
        # manifest's defaults — an asymmetry against the per-package function
        # above, and the mirror image of the one AGENTS.md already records for
        # `cargo_comment_audit.find_cargo_defaults` (which returns EMPTY
        # there). Neither behaviour is reachable: a `[features]` table with no
        # `[package]` is not valid cargo, and a virtual workspace manifest may
        # not carry features. Asserting either direction would cement an
        # arbitrary answer to a question no input can ask.
    return fails


def audit_repo_arms() -> list[str]:
    """End-to-end arms over the VERDICT, on fixture repos.

    Issue 790 T3. `selftest()` covers the reachability and closure SEMANTICS —
    the genuinely hard part, with fixtures taken from real workspace shapes —
    and `TOKENIZER_CASES` covers the line grammar. Neither reaches
    `audit_repo`, which is where those two models are joined into a verdict:
    17 of this module's 44 arm-reach survivors were in this function alone, and
    it takes a repo path, so the whole thing was armable the entire time.

    The classes below all fail SILENTLY: a wrong skip prints a smaller
    `checked` count and "0 mismatches", which is byte-identical to clean. That
    is the exact failure mode this file's own TOKENIZER_CASES comment describes
    ("riir-chain sat at '26 docs, 0 labels' unnoticed"), one layer up.

    ⚠ Every fixture manifest carries a `[package] name`: `reachable_features`
    resolves through the package graph and sees NOTHING without it (measured in
    `cargo_comment_audit._audit_repo_arms`, where three arms failed against
    correct code before that was understood).
    """
    import contextlib
    import io
    import tempfile

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    def run(manifests: dict[str, str], doc: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, (rel, body) in enumerate(manifests.items()):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not body.lstrip().startswith("[package]"):
                    body = f'[package]\nname = "fixture{i}"\n' + body
                p.write_text(body, encoding="utf-8")
            (root / ".benchmarks").mkdir()
            (root / ".benchmarks" / "b.md").write_text(doc, encoding="utf-8")
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), \
                 contextlib.redirect_stderr(io.StringIO()):
                n = audit_repo(root)
            return n, sink.getvalue()

    one = '[package]\nname = "demo"\n[features]\ndefault = ["alpha"]\nalpha = []\nbeta = []\n'

    # The two core mismatches, one per direction.
    n, out = run({"Cargo.toml": one}, "**Feature:** `beta` (**DEFAULT-ON**)\n")
    eq("a DEFAULT label on an opt-in feature is a MISMATCH", n, 1)
    eq("…and the label was counted as checked", "checked 1 labels" in out, True)
    n, _ = run({"Cargo.toml": one}, "**Feature:** `alpha` (opt-in)\n")
    eq("an OPT-IN label on a default-on feature is a MISMATCH", n, 1)

    # Both labels correct, and the population says so.
    n, out = run({"Cargo.toml": one},
                 "**Feature:** `alpha` (**DEFAULT-ON**)\n"
                 "**Feature:** `beta` (opt-in)\n")
    eq("correct labels are not a mismatch", n, 0)
    eq("both labels were checked", "checked 2 labels" in out, True)

    # FOREIGN: a `pkg/feat` qualifier that is not a crate here is unauditable,
    # and the skip must be VISIBLE. A silently dropped label is the failure
    # this whole script exists to catch (Issue 702).
    n, out = run({"Cargo.toml": one},
                 "**Feature:** `some-sibling/alpha` (opt-in)\n")
    eq("a foreign qualifier is skipped, not judged", n, 0)
    eq("…and the skip is REPORTED",
       "1 cross-repo (qualifier not a crate here)" in out, True)

    # A token that is not a defined Cargo feature is not a label at all — the
    # use-case / paper-artifact false-positive class.
    n, out = run({"Cargo.toml": one}, "**Feature:** `not_a_feature` (opt-in)\n")
    eq("an undefined feature name is not checked", (n, "checked 0 labels" in out),
       (0, True))

    # A line with no Feature header carries no labels, however it is written.
    n, out = run({"Cargo.toml": one}, "Some prose about `alpha` (opt-in).\n")
    eq("a non-header line is not scanned", (n, "checked 0 labels" in out),
       (0, True))

    # ── the qualified-DEFAULT-drift branch is guarded on `parsed == "default"`
    # (Issue 790 T2). The LAYER-SPLIT shape: the qualifier DEFINES the flag and
    # ships it OFF, while a DIFFERENT crate defaults the same NAME. A
    # **default** label there is drift — and an **opt-in** label is exactly
    # right, because the qualifier really does ship it off. Nothing reached
    # that conjunct: `qualified_default_drift` is armed on its own in
    # `selftest`, but no end-to-end fixture ever put a NON-default label in
    # front of it, so flipping the test to `!=` flagged correct opt-in labels
    # and every arm stayed green.
    split = {
        "engine/Cargo.toml":
            '[package]\nname = "riir-engine"\n'
            '[features]\ndefault = ["mop_runtime"]\nmop_runtime = []\n',
        "civ/Cargo.toml":
            '[package]\nname = "riir-games-civ"\n'
            '[features]\ndefault = []\nmop_runtime = []\n',
    }
    n, out = run(split, "**Feature:** `riir-games-civ/mop_runtime` (opt-in)\n")
    eq("a qualified OPT-IN label on a layer-split flag is correct, not drift",
       (n, "checked 1 labels" in out), (0, True))
    n, _ = run(split, "**Feature:** `riir-games-civ/mop_runtime` (**DEFAULT-ON**)\n")
    eq("…while the qualified DEFAULT label on the same flag IS drift", n, 1)

    # ── the mismatch REPORT line, on a QUALIFIED label (Issue 790 T2). Every
    # arm above reports through `{qual + '/' if qual else ''}` with qual=None,
    # so the conditional's live branch never ran and the same source line
    # survived in all three mismatch blocks. A reader acts on this line: it is
    # the only place the report says WHICH crate's flag is mislabelled.
    twocrate = {
        "a/Cargo.toml": '[package]\nname = "demo"\n'
                        '[features]\ndefault = ["alpha"]\nalpha = []\nbeta = []\n',
    }
    n, out = run(twocrate, "**Feature:** `demo/beta` (**DEFAULT-ON**)\n")
    eq("a qualified DEFAULT label on an opt-in flag names its qualifier",
       (n, "feat: demo/beta" in out), (1, True))
    n, out = run(twocrate, "**Feature:** `demo/alpha` (opt-in)\n")
    eq("a qualified OPT-IN label on a default flag names its qualifier",
       (n, "feat: demo/alpha" in out), (1, True))

    # SCOPED CLAIM: an explicit per-crate reading is trusted and counted.
    n, out = run({"Cargo.toml": one},
                 "**Feature:** `alpha` (opt-in in this crate)\n")
    eq("a crate-scoped opt-in claim is trusted",
       (n, "crate-scoped claim" in out), (0, True))

    # FORWARDED-ONLY: a qualified opt-in label whose QUALIFIER ships the flag
    # off is a defensible per-crate claim, even though another crate defaults
    # the same name (the Defense-3 layer split, riir-ai 681/905).
    split = {
        "Cargo.toml": '[package]\nname = "root"\n[features]\n'
                      'default = ["sub/alpha"]\nalpha = []\n',
        "crates/sub/Cargo.toml": '[package]\nname = "sub"\n[features]\n'
                                 'default = []\nalpha = []\n',
    }
    n, out = run(split, "**Feature:** `sub/alpha` (opt-in)\n")
    eq("a qualified opt-in scoped to a crate that ships it OFF is forwarded-only",
       (n, "forwarded-only (off in owning crate)" in out), (0, True))
    # …and the BARE form in the same repo is a DEPLOYED claim, which must still
    # red: a blanket forwarded-only rule silently excused two live labels here.
    n, _ = run(split, "**Feature:** `alpha` (opt-in)\n")
    eq("the BARE form of the same label is still a mismatch", n, 1)

    return fails


def audit_repo(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    print(f"\n=== Auditing {repo_root.name} ===")
    mans = load_manifests(repo_root)
    any_default = reachable_features(mans)
    # The (pkg, feat) node set behind any_default — the qualified-DEFAULT
    # counterpart adjudicates per-crate claims on NODES, not bare names.
    deployed_nodes = reachable_nodes(mans)
    defined_by_pkg = {p: set(f) for p, f in mans.items()}
    defined_features = find_defined_features(repo_root)
    package_names = find_package_names(repo_root)
    own_by_pkg = own_closures_by_pkg(repo_root)
    own_default = {f for s in own_by_pkg.values() for f in s}
    # own-crate default is a strict subset of the deployed default by
    # construction (a bare local entry is also an edge in the reachability
    # graph). If that ever fails the two models have diverged and the
    # forwarded-only suppression below would silently stop firing — so say so
    # rather than quietly mis-suppress. Holds in all 18 repos as of 2026-09-01.
    stray = own_default - any_default
    if stray:
        print(f"  WARN: own-default not a subset of deployed-default: "
              f"{sorted(stray)[:8]} ({len(stray)}) — the forwarded-only rule "
              f"may mis-suppress", file=sys.stderr)

    mismatches = 0
    checked = 0
    foreign = 0
    forwarded = 0
    scoped = 0
    qual_default = 0
    for rel, ln, line, qual, feat, raw, parsed in iter_bench_doc_labels(repo_root):
        # A `pkg/feature` label whose pkg is not a crate HERE is a sibling
        # repo's flag: unauditable from this repo, and auditing it against the
        # local manifests is actively wrong (see find_package_names). Count it
        # so the skip is visible — a silently dropped label is the failure mode
        # this whole script exists to catch (Issue 702).
        if qual is not None and qual not in package_names:
            foreign += 1
            continue
        # Only consider feature_name that exists as a defined Cargo feature.
        # This avoids false positives on use-case names, paper-artifact names,
        # etc. (the session-12 insight).
        if feat not in defined_features:
            continue
        checked += 1
        is_default = feat in any_default
        if parsed == "default" and not is_default:
            mismatches += 1
            print(f"  [MISMATCH] doc says DEFAULT but feature NOT in any Cargo default")
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {qual + '/' if qual else ''}{feat}  raw_status: {raw!r}")
            print(f"    line: {line}")
        elif (parsed == "default"
              and qualified_default_drift(qual, feat, own_by_pkg,
                                          defined_by_pkg, deployed_nodes)):
            # The qualified-DEFAULT counterpart: the qualifier DEFINES the
            # flag, ships it OFF, and the deployed on-ness belongs to a
            # DIFFERENT crate's same-named flag — the label attributes one
            # crate's gate state to another (the Defense-3 mirror; see
            # qualified_default_drift for the ancestor-forward shape this
            # deliberately does NOT fire on).
            qual_default += 1
            mismatches += 1
            print(f"  [MISMATCH] doc says DEFAULT but the QUALIFIER defines and "
                  f"ships the flag OFF (on only via another crate's "
                  f"same-named default)")
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {qual + '/' if qual else ''}{feat}  raw_status: {raw!r}")
            print(f"    line: {line}")
        elif parsed == "opt-in" and is_default:
            # Forwarded-only: the OWNING crate — for a qualified label, the
            # QUALIFIER — ships the flag off. "opt-in" is a defensible
            # reading for a per-crate claim, and the qualifier's own closure
            # is the model that decides it (see qualified_optin_is_scoped:
            # another crate's same-named default — the Defense-3 layer split
            # — does not falsify a crate-scoped claim; the riir-ai 681/905
            # false positives). A BARE token in a repo-level doc is making a
            # deployed claim, and suppressing it hides real drift:
            # katgpt-rs's own `` `still_kv` (opt-in, Plan 245) `` and
            # `` `hla_eigenbasis_recovery` (opt-in) `` are both reached from the
            # root default in three hops, and a blanket forwarded-only rule
            # silently excused both.
            if qualified_optin_is_scoped(qual, feat, own_by_pkg):
                forwarded += 1
                continue
            if SCOPED_CLAIM_RE.search(line):
                scoped += 1
                continue
            mismatches += 1
            print(f"  [MISMATCH] doc says OPT-IN but feature IS in some Cargo default")
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {qual + '/' if qual else ''}{feat}  raw_status: {raw!r}")
            print(f"    line: {line}")
    LABELS_CHECKED.append(checked)
    # ⛔ This line is an INTERFACE, not a message. `docs_drift_sweep.py` and
    # `.github/workflows/sibling_docs_drift.yml` both PARSE it, and adding the
    # `over N doc(s)` clause here broke the first — 0 labels in every repo, 9
    # false findings — while the second's looser `.*labels.*` survived.
    # Grep for `checked .* labels` before editing it.
    tail = (f"  -> checked {checked} labels over "
            f"{DOCS_WALKED.get(str(repo_root), 0)} doc(s), "
            f"{mismatches} mismatches")
    notes = []
    if foreign:
        notes.append(f"{foreign} cross-repo (qualifier not a crate here)")
    if forwarded:
        notes.append(f"{forwarded} forwarded-only (off in owning crate)")
    if scoped:
        notes.append(f"{scoped} crate-scoped claim")
    if qual_default:
        notes.append(f"{qual_default} qualified-default misattributed")
    untracked = UNTRACKED_SKIPPED.get(str(repo_root), 0)
    if untracked:
        notes.append(f"{untracked} untracked manifest(s) not in git")
    if notes:
        tail += " [skipped: " + "; ".join(notes) + "]"
    print(tail)
    return mismatches


# Real line shapes from the workspace, each with the (feature, status) the
# tokenizer must produce. This runs on EVERY invocation — the audit is wired
# per-push via scripts/docs_gate.sh, and a silent regex regression there does
# not fail anything: it just recognises fewer labels and still prints
# "0 mismatches". That is indistinguishable from clean, which is how riir-chain
# sat at "26 docs, 0 labels" unnoticed. A dropped shape must be loud.
# Each case is (line, expected_qualifier, expected_feature, expected_status).
TOKENIZER_CASES = [
    # katgpt-rs dialect
    ("**Feature:** `foo` (opt-in)", None, "foo", "opt-in"),
    ("**Feature:** `bar` (**DEFAULT-ON** since 2026-07-20)", None, "bar", "default"),
    # Namespaced `pkg/feature`. `/` was absent from the name class, so these
    # produced NO token at all: 15 in-repo labels across 3 repos were unread,
    # and reported as "0 mismatches" — clean and blind are the same output.
    ("**Feature:** `katgpt-core/gaussianity_probe` (opt-in)",
     "katgpt-core", "gaussianity_probe", "opt-in"),
    ("**Feature:** `riir-wallet/siwr` (**DEFAULT-ON**)",
     "riir-wallet", "siwr", "default"),
    # compound parenthetical: the status word is the LAST clause, and the tight
    # lazy capture stops at the first comma ("client + RP kit" -> unknown).
    ("**Features:** `riir-wallet/siwr` (client + RP kit, default-OFF), forwarded as",
     "riir-wallet", "siwr", "opt-in"),
    # riir-chain dialect: status bolded, then an unbracketed ` (` follows.
    # `(` and `*` were not status boundaries, so this yielded NO token at all.
    ("**Feature:** `baz` — **default-OFF** (implies `qux`)", None, "baz", "opt-in"),
    # negation must still win over the bare word "default"
    ("**Feature:** `neg` (opt-in, NOT default-on)", None, "neg", "opt-in"),
    # third state: no bare `default[]` entry, but a `pkg/feat` chain reaches it
    ("**Feature:** `tr` (opt-in at the time of this bench; **on by transitive "
     "default** as of 2026-09-01 — via `a/default -> b/tr`)", None, "tr", "default"),
    # ...and the negation of it must still read as opt-in
    ("**Feature:** `trneg` (opt-in, NOT on by transitive default)",
     None, "trneg", "opt-in"),
]


# The riir-engine shape that produced a confident false drift report. A future
# "simplification" back to `_parse_feature_spec` here re-introduces it, so it is
# pinned rather than merely commented.
# The two chains the (package, feature) graph exists for. Both were wrong under
# the models it replaced, in opposite directions, and both produced a confident
# audit verdict against a doc that was correct.
REACHABILITY_CASE = {
    # cross-manifest chain: civ/default -> osc_emotion -> shared/osc_emotion
    #                       -> osc_npc.  A per-manifest closure stops at the
    #                       manifest boundary and calls osc_npc opt-in.
    "riir-games-civ": {
        "default": ["osc_emotion"],
        "osc_emotion": ["civ_emotion", "riir-games-shared/osc_emotion"],
        "civ_emotion": [],
    },
    "riir-games-shared": {
        "osc_emotion": ["osc_npc"],
        "osc_npc": [],
    },
    # external edge: the target crate is NOT in this repo, so `tropical_algebra`
    # must NOT be credited. Collapsing `pkg/feat` to `feat` credits it.
    "riir-engine": {
        "default": ["se2_equivariant"],
        "se2_equivariant": ["katgpt-core/tropical_algebra"],
        "tropical_algebra": ["katgpt-core/tropical_algebra"],
    },
}


LOCAL_CLOSURE_CASE = {
    "default": ["se2_equivariant", "band_edge_trigger"],
    "se2_equivariant": ["katgpt-core/tropical_algebra"],
    "tropical_algebra": ["katgpt-core/tropical_algebra", "latent_functor"],
    "band_edge_trigger": ["hla"],
    "hla": [],
    "latent_functor": [],
}


def blindness_arms() -> list[str]:
    """`floor_verdict` and `BlindRead` — Issue 790 Finding 9's two halves.

    Both are unreachable from every other arm in this file: the floors are pin
    arithmetic outside the function that computes the number (Issue 775), and
    the abort only fires on a read that fails, which no fixture produces by
    accident.
    """
    import contextlib
    import io
    import tempfile

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    def quiet(root):
        """audit_repo writes its report to stdout; a fixture's copy of it in
        the middle of a real run is noise a reader has to learn to ignore."""
        with contextlib.redirect_stdout(io.StringIO()),              contextlib.redirect_stderr(io.StringIO()):
            return audit_repo(root)

    # ── the floors. Two, failing differently, and the WALK is checked first:
    # a dead walk makes the label count 0 too, and reporting that as a
    # tokenizer regression sends the reader to the wrong instrument.
    eq("a healthy self-audit clears both floors",
       floor_verdict(MIN_DOCS + 1, MIN_LABELS + 1), None)
    eq("exactly at the floors is clean, not a finding",
       floor_verdict(MIN_DOCS, MIN_LABELS), None)
    eq("a shrunken walk is reported as a WALK regression",
       "WALK went blind" in (floor_verdict(MIN_DOCS - 1, MIN_LABELS + 1) or ""),
       True)
    eq("a full walk with no labels is reported as READ or TOKENIZER",
       "READ or TOKENIZER" in (floor_verdict(MIN_DOCS + 1, MIN_LABELS - 1) or ""),
       True)
    eq("a dead walk is diagnosed as the WALK even though labels are 0 too",
       "WALK went blind" in (floor_verdict(0, 0) or ""), True)
    # The floors guard blindness, not churn: they must sit well below the
    # measured population or they red on ordinary doc edits.
    eq("the floors are below the 2026-09-15 measurement (439 docs / 97 labels)",
       (MIN_DOCS < 439, MIN_LABELS < 97), (True, True))

    # ── the abort. An OSError on a file the walk just listed is the
    # instrument going blind; a TOMLDecodeError is real content to walk past.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".benchmarks").mkdir()
        (root / ".benchmarks" / "b.md").write_text(
            "**Feature:** `f` (opt-in)\n", encoding="utf-8")
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\n[features]\ndefault = []\nf = []\n',
            encoding="utf-8")
        eq("the fixture repo audits cleanly before anything is broken",
           quiet(root), 0)

        # MALFORMED toml is a CONTENT problem: warn, walk past, keep auditing.
        (root / "Cargo.toml").write_text("[package\nname =", encoding="utf-8")
        try:
            quiet(root)
            eq("a malformed manifest does not abort the audit", True, True)
        except BlindRead:
            eq("a malformed manifest does not abort the audit", False, True)

        # UNREADABLE is the other case, in both directions.
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\n[features]\nf = []\n', encoding="utf-8")
        # The prefix match must be FORM-INDEPENDENT (Issue 812): audit_repo
        # resolves its root (`repo_root.resolve()`), so when TMPDIR's string
        # form is an unresolved symlink (/tmp → /private/tmp — reviewer box,
        # 2026-09-16) the fixture's spelling and the audit's disagree and the
        # patch silently stops matching: the arm then reports `got False` on
        # a healthy instrument. Accept EITHER spelling of the same root.
        root_forms = (str(root), str(root.resolve()))
        real_open, real_rt = Path.open, Path.read_text
        for attr, real, pred, who in (("open", real_open,
                                       lambda q: q.name == "Cargo.toml",
                                       "an unreadable MANIFEST"),
                                      ("read_text", real_rt,
                                       lambda q: q.suffix == ".md",
                                       "an unreadable DOC")):
            def boom(self, *a, _r=real, _p=pred, _forms=root_forms, **k):
                if _p(self) and any(
                        str(self).startswith(f) for f in _forms):
                    raise OSError(1450, "Insufficient system resources")
                return _r(self, *a, **k)

            setattr(Path, attr, boom)
            try:
                quiet(root)
                eq(f"{who} aborts rather than reporting a number", False, True)
            except BlindRead:
                eq(f"{who} aborts rather than reporting a number", True, True)
            finally:
                setattr(Path, attr, real)

    return fails


def audit_units(p: Path) -> list[Path]:
    """The directories ONE path argument expands to — each gets its own
    `=== Auditing <name> ===` block. Shared by both auditors' CLI and by
    `docs_drift_sweep.run_auditor`, which must attribute those blocks back to
    the repo it asked about: a repo with no root Cargo.toml is audited per
    child crate under the CHILD's name, and a sweep looking the repo up by its
    own name then silently read (0, 0) (measured 2026-10-27: reflex-site's
    only manifest is `wasm-head/Cargo.toml`). Sorted, so the block order is
    the same on every filesystem."""
    if (p / "Cargo.toml").exists():
        return [p]
    children = sorted(d for d in p.iterdir()
                      if d.is_dir() and (d / "Cargo.toml").exists())
    return children or [p]


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sub in ("rooted", "split/b", "split/a", "split/docs", "bare/docs"):
            (root / sub).mkdir(parents=True)
        for m in ("rooted/Cargo.toml", "split/b/Cargo.toml", "split/a/Cargo.toml"):
            (root / m).write_text("", encoding="utf-8")
        got = [[u.relative_to(root).as_posix() for u in audit_units(root / r)]
               for r in ("rooted", "split", "bare")]
        if got != [["rooted"], ["split/a", "split/b"], ["bare"]]:
            raise SystemExit(
                "✗ audit_units self-test FAILED\n"
                f"  got: {got}\n"
                "  a root manifest audits the repo itself; a manifest-less repo\n"
                "  audits each child crate, SORTED; a repo with neither audits\n"
                "  itself — the sweep attributes blocks by exactly this rule.")
    reach = reachable_features(REACHABILITY_CASE)
    if "osc_npc" not in reach or "tropical_algebra" in reach:
        raise SystemExit(
            "✗ reachability self-test FAILED\n"
            f"  got: {sorted(reach)}\n"
            "  `osc_npc` MUST be reachable (civ/default -> osc_emotion ->\n"
            "  riir-games-shared/osc_emotion -> osc_npc crosses a manifest\n"
            "  boundary), and `tropical_algebra` MUST NOT be (reachable only\n"
            "  via katgpt-core/, a crate outside the repo). Getting either wrong\n"
            "  reports drift on a doc that is correct — see find_cargo_defaults.")
    nodes = reachable_nodes(REACHABILITY_CASE)
    if (("riir-games-shared", "osc_npc") not in nodes
            or ("riir-engine", "tropical_algebra") in nodes):
        raise SystemExit(
            "✗ reachability-nodes self-test FAILED\n"
            f"  got: {sorted(nodes)}\n"
            "  (riir-games-shared, osc_npc) MUST be a reachable node and\n"
            "  (riir-engine, tropical_algebra) MUST NOT — the pair form is the\n"
            "  discriminator qualified_default_drift adjudicates on; if the\n"
            "  projection and the walk disagree, per-crate default claims are\n"
            "  judged against the wrong graph.")
    own = local_default_closure(LOCAL_CLOSURE_CASE)
    if "tropical_algebra" in own or "band_edge_trigger" not in own:
        raise SystemExit(
            "✗ own-default self-test FAILED\n"
            f"  got own-default closure: {sorted(own)}\n"
            "  `tropical_algebra` must NOT be in it (reachable only as\n"
            "  `katgpt-core/tropical_algebra`, a DIFFERENT crate's flag), and\n"
            "  `band_edge_trigger` MUST be (a bare entry in `default`).\n"
            "  Collapsing `pkg/feat` to `feat` here reports false drift on\n"
            "  correct docs — see local_default_closure.")
    # The layer-split shape (riir-ai 681/905, Plan 581 / Bench 906): the
    # qualifier's forward is opt-in while a DIFFERENT crate defaults the
    # same feature NAME. A qualified opt-in claim is scoped to its own
    # crate; a bare one stays a deployed claim; and a qualifier whose own
    # default DOES turn the flag on still flags.
    LAYER_SPLIT = {
        "riir-engine": {"mop_psafe", "mop_runtime", "mop_homeostasis"},
        "riir-games-civ": set(),
    }
    if not qualified_optin_is_scoped("riir-games-civ", "mop_runtime", LAYER_SPLIT):
        raise SystemExit(
            "✗ layer-split self-test FAILED: a qualified opt-in label whose\n"
            "  qualifier ships the flag off must be scoped-skipped even when\n"
            "  another crate defaults the same name (the Defense-3 split,\n"
            "  riir-ai .benchmarks/681 + 905 — false drift on correct docs)")
    if qualified_optin_is_scoped("riir-engine", "mop_runtime", LAYER_SPLIT):
        raise SystemExit(
            "✗ layer-split self-test FAILED: when the QUALIFIER's own default\n"
            "  turns the flag on, its opt-in label is real drift and must NOT\n"
            "  be scoped-skipped")
    if qualified_optin_is_scoped(None, "mop_runtime", LAYER_SPLIT):
        raise SystemExit(
            "✗ layer-split self-test FAILED: a BARE label is a deployed\n"
            "  claim — the union model governs, never the qualifier escape")
    # The qualified-DEFAULT counterpart, pinned in BOTH directions. Same
    # Defense-3 shape, read backwards: a doc claiming the qualifier's flag is
    # DEFAULT because a DIFFERENT crate promoted the same NAME misattributes
    # one crate's gate state to another and MUST flag — while the honest
    # shapes (qualifier defaults it; ancestor forwards THE QUALIFIER's own
    # node; bare label; qualifier citing a dependency's flag) MUST NOT.
    LAYER_SPLIT_DEFINED = {
        "riir-engine": {"mop_psafe", "mop_runtime", "mop_homeostasis"},
        "riir-games-civ": {"mop_runtime", "mop_homeostasis"},
    }
    LAYER_SPLIT_NODES = {
        ("riir-engine", "mop_psafe"),
        ("riir-engine", "mop_runtime"),
        ("riir-engine", "mop_homeostasis"),
    }
    if not qualified_default_drift("riir-games-civ", "mop_runtime", LAYER_SPLIT,
                                   LAYER_SPLIT_DEFINED, LAYER_SPLIT_NODES):
        raise SystemExit(
            "✗ qualified-default self-test FAILED: a DEFAULT label on a\n"
            "  qualifier that DEFINES the flag and ships it OFF, when the\n"
            "  deployed on-ness belongs to a different crate's same-named\n"
            "  flag, is misattribution and MUST flag (the Defense-3 mirror\n"
            "  of the riir-ai 681/905 lesson — this direction used to pass\n"
            "  silently through the bare-name union)")
    if qualified_default_drift("riir-engine", "mop_runtime", LAYER_SPLIT,
                               LAYER_SPLIT_DEFINED, LAYER_SPLIT_NODES):
        raise SystemExit(
            "✗ qualified-default self-test FAILED: the qualifier's own default\n"
            "  turns the flag on — its DEFAULT label holds and must NOT flag")
    if qualified_default_drift(None, "mop_runtime", LAYER_SPLIT,
                               LAYER_SPLIT_DEFINED, LAYER_SPLIT_NODES):
        raise SystemExit(
            "✗ qualified-default self-test FAILED: a BARE default label is a\n"
            "  deployed claim — the union model governs, never this detector")
    # The ancestor-forward shape (riir-chain siwr, 2026-09-01): the qualifier
    # defines + ships the flag off, but an ancestor's default forwards THE
    # QUALIFIER'S OWN (pkg, feat) node. Both readings are defensible there —
    # the node test is exactly what tells this apart from misattribution.
    FWD_OWN = {"riir-wallet": set()}
    FWD_DEFINED = {"riir-wallet": {"siwr"}}
    FWD_NODES = {("riir-wallet", "siwr")}
    if qualified_default_drift("riir-wallet", "siwr", FWD_OWN, FWD_DEFINED,
                              FWD_NODES):
        raise SystemExit(
            "✗ qualified-default self-test FAILED: the forwarded-only shape —\n"
            "  the qualifier's own (pkg, feat) node IS reachable from a\n"
            "  default, so a DEFAULT label is a defensible deployed reading\n"
            "  (the siwr doctrine) and must NOT flag")
    # A qualifier that does not DEFINE the flag is citing a dependency's
    # flag; the union model already adjudicated that claim — not this
    # detector's to re-judge.
    if qualified_default_drift("riir-games-civ", "osc_npc", LAYER_SPLIT,
                               LAYER_SPLIT_DEFINED, LAYER_SPLIT_NODES):
        raise SystemExit(
            "✗ qualified-default self-test FAILED: the qualifier does not\n"
            "  define this flag — the claim is a deployed citation of a\n"
            "  dependency's flag, already judged by the union model")
    for line, want_qual, want_feat, want_status in TOKENIZER_CASES:
        assert FEATURE_HEADER_RE.match(line), f"header regex missed: {line!r}"
        got = [(m.group("qual"), m.group("feat"), classify_token(line, m)[0])
               for m in FEATURE_TOKEN_RE.finditer(line)]
        if (want_qual, want_feat, want_status) not in got:
            raise SystemExit(
                f"✗ tokenizer self-test FAILED\n  line:   {line!r}\n"
                f"  want:   {(want_qual, want_feat, want_status)}\n  got:    {got}\n"
                "  A shape this script used to recognise no longer parses. It "
                "would keep printing '0 mismatches' over fewer labels.")
    # The VERDICT, end to end (Issue 790 T3). Everything above tests the two
    # MODELS; this tests the function that joins them, which is where 17 of
    # this module's arm-reach survivors were.
    verdict = audit_repo_arms()
    if verdict:
        raise SystemExit("✗ audit_repo self-test FAILED — the verdict logic that "
                         "joins the reachability model to the tokenizer does not "
                         "behave as documented:\n" + "\n".join(verdict))
    # The PURE rules and the manifest READERS (Issue 790 T2). Everything above
    # tests the two models and their join; these are the guards that decide
    # what enters a model in the first place, and each fails by quietly
    # shrinking the population rather than by saying anything.
    rules = pure_rule_arms() + manifest_reader_arms() + blindness_arms()
    if rules:
        raise SystemExit("✗ rule/reader self-test FAILED — a guard that decides "
                         "which packages and labels enter the models does not "
                         "behave as documented:\n" + "\n".join(rules))


# ── The blindness floors, katgpt-rs-scoped ─────────────────────────────────
# Two, and neither is redundant, because they go blind at different stages and
# only one of them moves when a tokenizer breaks on an unchanged tree:
#   MIN_DOCS    the WALK. `.benchmarks`/`.docs` renamed, a `rglob` regression,
#               a checkout that did not land — the label count goes to 0 and
#               "0 labels, 0 mismatches" is byte-identical to clean.
#   MIN_LABELS  the READ + the TOKENIZER over a walk that is still full size.
#               This is riir-chain's "26 docs, 0 labels" shape.
#
# ⚠ They bound the TOTAL failures and NOT the partial one. Measured (Issue 790
# Finding 9): a 50%-failing manifest read keeps the label count at its full 97
# and prints 24 fabricated mismatches — above both floors, every time. That
# half is `BlindRead`'s, not these numbers'. A floor is a blindness detector,
# never a correctness proof, and pairing the two was the point of measuring
# rather than inferring.
#
# Measured 2026-09-15: 439 docs, 97 labels. Floored well below, because these
# must red on an instrument going blind and NOT on ordinary doc churn.
MIN_DOCS = 250
MIN_LABELS = 50


def floor_verdict(n_docs: int, n_labels: int) -> str | None:
    """The blindness verdict for a self-audit, or None. Pure, so it is armable
    without a tree — the Issue 775 rule that keeps pin arithmetic out of main().
    """
    if n_docs < MIN_DOCS:
        return (f"walked {n_docs} doc(s), floor {MIN_DOCS} — the WALK went "
                f"blind; a smaller corpus still prints '0 mismatches'")
    if n_labels < MIN_LABELS:
        return (f"checked {n_labels} label(s) over {n_docs} doc(s), floor "
                f"{MIN_LABELS} — the walk is full size and the labels are "
                f"gone, so this is a READ or TOKENIZER regression")
    return None


def main(argv: list[str]) -> int:
    # The guard wraps this function's WHOLE body rather than delegating to a
    # `_main`, deliberately: `arm_reach_audit.EXEMPT_FUNCTIONS` exempts the CLI
    # shell by the name `main`, so splitting the dispatch into a differently
    # named helper moves argv handling out from under that pin and reports four
    # phantom survivors. Widening a vocabulary shared by every module in the
    # workspace to accommodate one refactor is the permissive direction; this
    # costs one indent level.
    #
    # selftest() is INSIDE the guard: on the starved box that motivated this,
    # its own fixture reads fail too, and an uncaught traceback there says
    # nothing about which of the two failures the reader should act on.
    try:
        selftest()
        if len(argv) < 2:
            # Default: audit the repo this script lives in. This is the
            # docs_gate invocation, and the only one the floors calibrate for.
            here = Path(__file__).resolve().parent.parent
            n = audit_repo(here)
            blind = floor_verdict(DOCS_WALKED.get(str(here), 0),
                                  LABELS_CHECKED[-1])
            if blind:
                print(f"\u26d4 bench_doc_audit BLINDNESS FLOOR \u2014 {blind}",
                      file=sys.stderr)
                return 2
            return 1 if n else 0
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
    except BlindRead as e:
        # Exit 2, never 1: `1` is this script's "mismatches were found", and an
        # instrument that could not READ has found nothing at all. The two must
        # not share an exit code \u2014 docs_gate.sh reports on the code alone.
        print(f"\u26d4 bench_doc_audit ABORTED \u2014 the tree is unreadable, "
              f"so every verdict below it would be computed against a truncated "
              f"model:\n    {e}\n  This is NOT '0 mismatches'. Re-run on a box "
              f"with resources; see BlindRead's docstring for the measurement.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
