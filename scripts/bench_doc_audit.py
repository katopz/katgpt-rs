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


def reachable_features(mans: dict[str, dict]) -> set[str]:
    """The `(package, feature)` reachability walk — pure, so it can be pinned."""
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
    return {f for _, f in seen if f != "default"}


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

    Note the unhandled counterpart, deliberately: a QUALIFIED label claiming
    DEFAULT still passes via `any_default` even when the qualifier ships the
    flag off. No such label exists in the workspace today (measured
    2026-09-12); tightening that direction changes green verdicts and is not
    done blind.
    """
    return qual is not None and feat not in own_by_pkg.get(qual, set())


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
    """Yield (rel, lineno, line, qualifier|None, feature, raw, parsed)."""
    for sub in (".benchmarks", ".docs"):
        d = repo_root / sub
        if not d.is_dir():
            continue
        for md in d.rglob("*.md"):
            rel = md.relative_to(repo_root).as_posix()
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
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


def audit_repo(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    print(f"\n=== Auditing {repo_root.name} ===")
    any_default = find_cargo_defaults(repo_root)
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
    tail = f"  -> checked {checked} labels, {mismatches} mismatches"
    notes = []
    if foreign:
        notes.append(f"{foreign} cross-repo (qualifier not a crate here)")
    if forwarded:
        notes.append(f"{forwarded} forwarded-only (off in owning crate)")
    if scoped:
        notes.append(f"{scoped} crate-scoped claim")
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


def selftest() -> None:
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


def main(argv: list[str]) -> int:
    selftest()
    if len(argv) < 2:
        # Default: audit the repo this script lives in.
        here = Path(__file__).resolve().parent.parent
        return 1 if audit_repo(here) else 0
    total = 0
    for arg in argv[1:]:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            print(f"skip (not a dir): {arg}", file=sys.stderr)
            continue
        if (p / "Cargo.toml").exists():
            total += audit_repo(p)
        else:
            children = [
                d for d in p.iterdir() if d.is_dir() and (d / "Cargo.toml").exists()
            ]
            if children:
                for c in children:
                    total += audit_repo(c)
            else:
                total += audit_repo(p)
    print(f"\n=== TOTAL mismatches across all repos: {total} ===")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
